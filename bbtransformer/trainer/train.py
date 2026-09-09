# bbtransformer\trainer\train.py

# Import Essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from pytorch_optimizer import Ranger21
import numpy as np
from tqdm import tqdm
from typing import Any


# ======================
# LOSS FUNCTION 
# ======================

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, momentum: float = 0.99, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('p_pos', torch.tensor(0.5))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        if self.training:
            with torch.no_grad():
                current_p_pos = probs.mean()
                if torch.isfinite(current_p_pos):
                    self.p_pos = self.momentum * self.p_pos + (1 - self.momentum) * current_p_pos

        alpha = 1.0 - self.p_pos
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ======================
# TRAINING FUNCTION
# ======================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 10000,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 100,
    use_focal_loss: bool = False,
    early_stop_metric: str = "f1"
):
    """
    Train BBTransformer with Ranger21 and adaptive focal loss.
    Early stopping based on validation metric: 'f1' (default) or 'loss'.
    """
    if early_stop_metric not in {"f1", "loss"}:
        raise ValueError("early_stop_metric must be 'f1' or 'loss'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Ranger21 with internal LR scheduling
    total_steps = epochs * len(train_loader)
    optimizer = Ranger21(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        num_iterations=total_steps,
        disable_lr_scheduler=False
    )

    criterion = AdaptiveFocalLoss(gamma=2.0, momentum=0.99) if use_focal_loss else nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # Early stopping state
    best_score = -float('inf') if early_stop_metric == "f1" else float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # ---------------------
        # Training
        # ---------------------
        model.train()
        for fmri, age, ext, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            fmri = fmri.to(device, non_blocking=True)
            age = age.to(device, non_blocking=True)
            ext = ext.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                logits = model(fmri, age, ext)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # ---------------------
        # Validation
        # ---------------------
        model.eval()
        all_probs, all_targets = [], []
        total_val_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for fmri, age, ext, labels in val_loader:
                fmri = fmri.to(device, non_blocking=True)
                age = age.to(device, non_blocking=True)
                ext = ext.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()

                with autocast(device_type=device.type):
                    logits = model(fmri, age, ext)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item() * labels.size(0)
                    n_samples += labels.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.ravel())
                all_targets.extend(labels.cpu().numpy())

        val_loss = total_val_loss / n_samples if n_samples > 0 else float('inf')

        # Compute F1 only if needed
        if early_stop_metric == "f1":
            preds = (np.array(all_probs) > 0.5).astype(int)
            val_score = f1_score(all_targets, preds, zero_division=0)
        else:  # "loss"
            val_score = val_loss

        # ---------------------
        # Early Stopping
        # ---------------------
        is_better = (val_score > best_score) if early_stop_metric == "f1" else (val_score < best_score)
        if is_better:
            best_score = val_score
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                metric_name = "F1" if early_stop_metric == "f1" else "Loss"
                print(f"Early stopping at epoch {epoch+1} ({metric_name}: {val_score:.4f})")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    return model

