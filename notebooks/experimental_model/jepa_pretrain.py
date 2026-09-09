# jepa_pretrain.py

"""
State-of-the-Art JEPA Pretraining for BBTransformer (GPU-Optimized & Safe)
======================================================================
Aligned with I-JEPA/V-JEPA references + Production GPU Fixes:
1. Shared mask token + positional embedding (I-JEPA)
2. Positional encoding in predictor cross-attention (V-JEPA)
3. Encode ALL patches before gathering (fixes RoPE corruption)
4. Pass context_indices to predictor
5. VECTORIZED Random Masking (Eliminates CPU bottleneck)
6. Smooth L1 loss + VICReg variance regularization (NO covariance term)
7. Cosine EMA schedule (I-JEPA/BYOL/DINO)
8. Linear LR warmup + cosine decay (I-JEPA recipe)
9. OPTIMIZED DataLoader (pin_memory, num_workers, persistent_workers)

NOTE: VICReg covariance term intentionally REMOVED. Brain features
 are inherently correlated — forcing orthogonality destroys the functional 
 network signals the downstream classifier needs.
"""

import os
import copy
import math
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Assuming bbtransformer is installed or in the python path
from bbtransformer import BBTransformer, create_bbtransformer

# =============================================================================
# NAMED CONSTANTS (No magic numbers)
# =============================================================================

# BBTransformer Architecture Defaults
FEATURE_DIM: int = 414
SEQ_LEN: int = 490
EMBED_DIM: int = 512
NUM_HEADS: int = 16
NUM_KV_HEADS: int = 8
NUM_LAYERS: int = 7

# JEPA Specific Constants
MAX_PATCHES: int = SEQ_LEN  # Predictor positional embedding table size
PREDICTOR_DEPTH: int = 4
PREDICTOR_DROPOUT: float = 0.1
MASK_RATIO: float = 0.5
MOMENTUM_START: float = 0.996
MOMENTUM_END: float = 0.999
WARMUP_FRACTION: float = 0.1
LEARNING_RATE: float = 1e-4  # Lower LR for transformers compared to standard CNNs/MLPs
BETAS: Tuple[float, float] = (0.9, 0.95)
WEIGHT_DECAY: float = 1e-5
GRAD_CLIP_NORM: float = 1.0
SMOOTH_L1_BETA: float = 1.0
VAR_LOSS_WEIGHT: float = 0.1
VARIANCE_EPS: float = 1e-4

# Training Constants
LOG_INTERVAL: int = 10
NUM_WORKERS: int = 4
BATCH_SIZE: int = 64  # Adjusted for fMRI memory footprint
EPOCHS: int = 300
DEFAULT_SAVE_PATH: str = "weights/jepa_bbtransformer.pth"

# Keys to extract for downstream fine-tuning
ENCODER_KEYS: frozenset = frozenset({"input_norm", "input_proj", "layers"})

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# JEPAPredictor
# =============================================================================

class JEPAPredictor(nn.Module):
    """
    Full attention-based predictor matching I-JEPA/V-JEPA architecture.
    Uses a single shared mask token conditioned on positional embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        seq_len: int,
        predictor_depth: int = PREDICTOR_DEPTH,
        dropout: float = PREDICTOR_DROPOUT,
    ) -> None:
        super().__init__()
        assert embed_dim > 0 and num_heads > 0 and embed_dim % num_heads == 0
        assert seq_len > 0 and predictor_depth > 0
        assert 0.0 <= dropout < 1.0

        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Shared mask token (I-JEPA reference)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding tables
        self.target_pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.context_pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q": nn.RMSNorm(embed_dim),
                "norm_kv": nn.RMSNorm(embed_dim),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim, num_heads=num_heads,
                    batch_first=True, dropout=dropout,
                ),
                "norm_ffn": nn.RMSNorm(embed_dim),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4, bias=False),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim, bias=False),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(predictor_depth)
        ])

        self.output_norm = nn.RMSNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        context_encoded: torch.Tensor,
        mask_indices: torch.Tensor,
        context_indices: torch.Tensor,
    ) -> torch.Tensor:
        B, N_mask = mask_indices.shape
        device = context_encoded.device

        # Shared mask token + positional embedding
        shared_token = self.mask_token.expand(B, N_mask, -1)
        tgt_pos = self.target_pos_embed.expand(B, -1, -1)
        tgt_idx_exp = mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        target_pos = torch.gather(tgt_pos, 1, tgt_idx_exp)
        queries = shared_token + target_pos

        # Add positional info to context tokens
        ctx_pos = self.context_pos_embed.expand(B, -1, -1)
        ctx_idx_exp = context_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        context_pos = torch.gather(ctx_pos, 1, ctx_idx_exp)
        context_with_pos = context_encoded + context_pos

        # Cross-attention refinement
        x = queries
        for layer in self.cross_attn_layers:
            q = layer["norm_q"](x)
            kv = layer["norm_kv"](context_with_pos)
            attn_out, _ = layer["cross_attn"](q, kv, kv)
            x = x + attn_out
            x = x + layer["ffn"](layer["norm_ffn"](x))

        return self.output_proj(self.output_norm(x))


# =============================================================================
# JEPA Wrapper
# =============================================================================

class JEPAWrapper(nn.Module):
    """Wraps BBTransformer with JEPA pretraining logic."""

    def __init__(
        self,
        base_model: BBTransformer,
        embed_dim: int,
        seq_len: int,
        mask_ratio: float = MASK_RATIO,
        num_heads: int = NUM_HEADS,
    ) -> None:
        super().__init__()
        assert isinstance(base_model, BBTransformer)
        assert 0.0 < mask_ratio < 1.0

        self.base_model = base_model
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Target encoder (frozen deep copy of backbone only)
        self.target_input_norm = copy.deepcopy(base_model.input_norm)
        self.target_input_proj = copy.deepcopy(base_model.input_proj)
        self.target_layers = copy.deepcopy(base_model.layers)

        for module in [self.target_input_norm, self.target_input_proj, self.target_layers]:
            for p in module.parameters():
                p.requires_grad = False

        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            predictor_depth=PREDICTOR_DEPTH,
            dropout=PREDICTOR_DROPOUT,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Update target encoder via exponential moving average."""
        assert 0.0 <= momentum <= 1.0

        pairs = [
            (self.target_input_norm, self.base_model.input_norm),
            (self.target_input_proj, self.base_model.input_proj),
        ]
        for tgt_layer, ctx_layer in zip(self.target_layers, self.base_model.layers):
            pairs.append((tgt_layer, ctx_layer))

        for tgt_mod, ctx_mod in pairs:
            for pt, pc in zip(tgt_mod.parameters(), ctx_mod.parameters()):
                pt.data.mul_(momentum).add_(pc.data, alpha=1.0 - momentum)

    def _run_layer(self, layer: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
        """Execute a single BBTransformer layer block (Pre-Norm Residual)."""
        # Attention block
        residual = x
        x = layer['norm1'](x)
        x = layer['attn'](x)
        x = layer['drop_path1'](x)
        x = residual + x
        x = layer['norm_post_attn'](x)

        # FFN block
        residual = x
        x = layer['norm2'](x)
        x = layer['ffn'](x)
        x = layer['dropout_ffn'](x)
        x = layer['drop_path2'](x)
        x = residual + x
        x = layer['norm_post_ffn'](x)
        return x

    def _generate_masks(
        self, B: int, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate vectorized random masks for the sequence."""
        n_mask = int(seq_len * self.mask_ratio)
        
        # Generate random permutations for each batch item
        perms = torch.stack([torch.randperm(seq_len, device=device) for _ in range(B)])
        
        mask_indices = perms[:, :n_mask]
        context_indices = perms[:, n_mask:]
        
        return mask_indices, context_indices

    def forward_pretrain(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute one JEPA pretraining forward pass."""
        assert x.dim() == 3, f"Input must be 3D (B, T, D), got shape {x.shape}"
        B, T, D = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"
        assert torch.isfinite(x).all(), "Input contains NaN or Inf values"

        was_training = self.base_model.training
        self.base_model.eval()  # Disable dropout/stochastic depth for consistent targets

        try:
            # 1. Context Encoder (processes full sequence to preserve RoPE positions)
            ctx_x = self.base_model.input_norm(x)
            ctx_proj = self.base_model.input_proj(ctx_x)

            all_encoded = ctx_proj
            for layer in self.base_model.layers:
                all_encoded = self._run_layer(layer, all_encoded)

            # 2. Generate Masks
            mask_indices, context_indices = self._generate_masks(B, T, x.device)

            # 3. Gather visible patches for Predictor
            ctx_idx_exp = context_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            context_encoded = torch.gather(all_encoded, 1, ctx_idx_exp)

            # 4. Target Encoder (no gradients, processes full sequence)
            with torch.no_grad():
                tgt_x = self.target_input_norm(x)
                tgt_proj = self.target_input_proj(tgt_x)

                target_encoded = tgt_proj
                for layer in self.target_layers:
                    target_encoded = self._run_layer(layer, target_encoded)

                mask_idx_exp = mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
                target_repr = torch.gather(target_encoded, 1, mask_idx_exp)

            # 5. Predict
            predicted = self.predictor(context_encoded, mask_indices, context_indices)

            # 6. Losses
            pred_loss = F.smooth_l1_loss(predicted, target_repr, beta=SMOOTH_L1_BETA)

            # VICReg Variance Loss (prevents representation collapse)
            std_pred = torch.sqrt(predicted.var(dim=0) + VARIANCE_EPS)
            std_tgt = torch.sqrt(target_repr.var(dim=0) + VARIANCE_EPS)
            var_loss = F.relu(1.0 - std_pred).mean() + F.relu(1.0 - std_tgt).mean()

            total_loss = pred_loss + VAR_LOSS_WEIGHT * var_loss

            assert torch.isfinite(total_loss), "Total loss is NaN or Inf"

            return {
                "loss": total_loss,
                "pred_loss": pred_loss,
                "var_loss": var_loss,
            }
        finally:
            if was_training:
                self.base_model.train()

    def extract_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """Extract only encoder backbone weights compatible with strict=False loading."""
        state: Dict[str, torch.Tensor] = {}
        for k, v in self.base_model.state_dict().items():
            top_level = k.split(".")[0]
            if top_level in ENCODER_KEYS:
                state[k] = v.cpu()

        assert len(state) > 0, "No encoder weights extracted. Check ENCODER_KEYS."
        logger.info("Extracted %d encoder backbone keys", len(state))
        return state


# =============================================================================
# Data Loading
# =============================================================================

def load_fmri_data(data_path: str) -> torch.Tensor:
    """
    Load unlabeled fMRI windows from .npz file for JEPA pretraining.
    
    Args:
        data_path: Path to the .npz file containing 'data' array.
        
    Returns:
        Tensor of shape (N_subjects, T, D).
    """
    assert os.path.exists(data_path), f"fMRI data not found: {data_path}"
    
    npz = np.load(data_path)
    assert "data" in npz, "NPZ file must contain 'data' array"
    
    fmri = npz["data"].astype(np.float32)
    assert fmri.ndim == 3, f"Expected 3D fMRI data (N, T, D), got shape {fmri.shape}"
    
    # Handle NaNs/Infs safely
    fmri = np.nan_to_num(fmri, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_tensor = torch.from_numpy(fmri)
    logger.info(
        "Loaded fMRI data: %d subjects, %d timepoints, %d features",
        X_tensor.size(0), X_tensor.size(1), X_tensor.size(2)
    )
    return X_tensor


# =============================================================================
# Main Pretraining Function
# =============================================================================

def pretrain_jepa(
    data_path: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    mask_ratio: float = MASK_RATIO,
    momentum_start: float = MOMENTUM_START,
    momentum_end: float = MOMENTUM_END,
    warmup_fraction: float = WARMUP_FRACTION,
    save_path: str = DEFAULT_SAVE_PATH,
) -> str:
    """Run JEPA pretraining and save encoder weights."""
    assert os.path.exists(data_path)
    assert epochs > 0 and batch_size > 0 and lr > 0
    assert 0.0 < mask_ratio < 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    logger.info("Device: %s | AMP: %s", device, use_amp)

    # Load data
    X_unlabeled = load_fmri_data(data_path)
    seq_len = X_unlabeled.size(1)
    feature_dim = X_unlabeled.size(2)

    config = dict(
        feature_dim=feature_dim,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        num_layers=NUM_LAYERS,
        # Add other BBTransformer specific configs if overriding defaults
    )

    # GPU-optimized DataLoader
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS if use_amp else 0,
        pin_memory=use_amp,
        persistent_workers=(use_amp and NUM_WORKERS > 0),
        drop_last=True,
    )

    # TensorDataset with single tensor yields tuples of length 1
    loader = DataLoader(TensorDataset(X_unlabeled), **loader_kwargs)
    assert len(loader) > 0, "DataLoader produced zero batches."

    base_model = create_bbtransformer(config).to(device)
    jepa = JEPAWrapper(
        base_model, 
        embed_dim=EMBED_DIM, 
        seq_len=seq_len,
        mask_ratio=mask_ratio, 
        num_heads=NUM_HEADS,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, jepa.parameters()),
        lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler(enabled=use_amp)

    warmup_epochs = int(epochs * warmup_fraction)

    logger.info(
        "Starting JEPA pretraining: %d epochs, %d subjects, %d batches/epoch",
        epochs, len(X_unlabeled), len(loader),
    )
    
    jepa.train()

    for epoch in tqdm(range(epochs), desc="JEPA Pretraining"):
        # Linear LR warmup then cosine decay
        if epoch < warmup_epochs:
            current_lr = lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Cosine EMA schedule
        momentum = momentum_end - (momentum_end - momentum_start) * (
            (1 + math.cos(math.pi * epoch / epochs)) / 2
        )

        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                out = jepa.forward_pretrain(xb)

            scaler.scale(out["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(jepa.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            jepa.update_target_encoder(momentum=momentum)
            epoch_loss += out["loss"].item()

        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % LOG_INTERVAL == 0:
            logger.info(
                "Epoch %d/%d | Loss: %.4f | LR: %.6f | EMA: %.4f",
                epoch + 1, epochs, avg_loss, current_lr, momentum,
            )

    # Save encoder weights
    weights = jepa.extract_encoder_weights()
    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    torch.save(weights, save_path)

    assert os.path.exists(save_path), f"Failed to save weights to {save_path}"
    logger.info("Saved JEPA pretrained encoder to %s (%d keys)", save_path, len(weights))
    return save_path
