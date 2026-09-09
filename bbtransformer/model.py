# model.py 

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any, List


# ======================
# TRANSFORMER COMPONENTS
# ======================

class RMSNorm(nn.Module):
    """RMSNorm implementation - no need for FP32 casting"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings with caching"""
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cache = {}

    def get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or compute rotary embeddings with caching"""
        cache_key = (seq_len, str(device))
        if cache_key not in self._cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cache[cache_key] = (emb.cos(), emb.sin())
        return self._cache[cache_key]

    def forward(self, seq_len: int, device: torch.device):
        return self.get_cos_sin(seq_len, device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to q and k"""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DropPath(nn.Module):
    """Stochastic Depth / Drop Path with proper scaling"""
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        if self.scale_by_keep:
            output = x.div(keep_prob) * random_tensor
        else:
            output = x * random_tensor
        return output


# ======================
# PATCH EMBEDDING
# ======================
class PatchEmbedding(nn.Module):
    """Patch embedding for time series data"""
    def __init__(self, patch_size: int, in_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_dim, embed_dim, bias=False)
        self.norm = RMSNorm(embed_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # Handle sequences not divisible by patch_size
        if T % self.patch_size != 0:
            trim = T % self.patch_size
            x = x[:, :-trim]
            T = x.shape[1]
        
        # Reshape into patches and project
        x = x.reshape(B, T // self.patch_size, self.patch_size * D)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class GQAWithRoPE(nn.Module):
    """Grouped-Query Attention with RoPE (uses PyTorch's optimized attention)"""
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        n_kv_heads: Optional[int] = None, 
        dropout: float = 0.1, 
        return_attn_weights: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or max(1, n_heads // 4)
        self.return_attn_weights = return_attn_weights
        
        # Ensure n_kv_heads is valid
        if self.n_kv_heads > n_heads:
            self.n_kv_heads = n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Rotary embeddings
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand k/v to match q heads (GQA)
        if self.n_heads != self.n_kv_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        # Compute attention
        if self.return_attn_weights:
            # Manual attention for weight extraction (needed for interpretability)
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            if self.training and self.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            attn_output = torch.matmul(attn_weights, v)
            self.last_attn_weights = attn_weights.detach()
        else:
            # PyTorch's optimized attention (automatically uses FlashAttention when available)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
            self.last_attn_weights = None
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)


# ======================
#  MODEL ARCHITECTURE
# ======================
class BBTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 1,
        embed_dim: int = 512,
        num_heads: int = 16,
        num_layers: int = 7,
        dropout_input: float = 0.18,
        dropout_patch: float = 0.16,
        dropout_attn: float = 0.15,
        dropout_ffn: float = 0.25,
        dropout_classifier: float = 0.07,
        dropout_temporal: float = 0.16,
        embed_dim_age: int = 32,
        embed_dim_ext: int = 16,
        patch_size: int = 3,
        patch_embed_ratio: float = 0.75,
        temp_attn_hidden: int = 512,
        n_kv_heads: Optional[int] = 4,
        return_attn_weights: bool = False,
        stochastic_depth_rate: float = 0.07,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed_dim = int(embed_dim * patch_embed_ratio)
        self.return_attn_weights = return_attn_weights
        self.num_layers = num_layers
        self.stochastic_depth_rate = stochastic_depth_rate

        # Input normalization and projection
        self.input_norm = RMSNorm(feature_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim, bias=False),
            RMSNorm(embed_dim),
            nn.Dropout(dropout_input)
        )

        # Patch embedding (short-scale)
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_dim=embed_dim,
            embed_dim=self.patch_embed_dim,
            dropout=dropout_patch
        )

        # Temporal attention for global pooling
        self.temporal_attn = nn.Sequential(
            nn.Linear(embed_dim, temp_attn_hidden, bias=True),
            nn.Tanh(),
            nn.Dropout(dropout_temporal),
            nn.Linear(temp_attn_hidden, 1, bias=True)
        )
        
        self.last_attention_maps = {}

        # GQA configuration
        if n_kv_heads is None:
            n_kv_heads = num_heads // 4 if num_heads >= 8 else max(1, num_heads // 2)
        self.n_kv_heads = n_kv_heads

        # Transformer layers with stochastic depth
        self.layers = nn.ModuleList()
        dpr = [stochastic_depth_rate * i / max(1, num_layers - 1) for i in range(num_layers)]
        
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'norm1': RMSNorm(embed_dim),
                'attn': GQAWithRoPE(
                    d_model=embed_dim,
                    n_heads=num_heads,
                    n_kv_heads=n_kv_heads,
                    dropout=dropout_attn,
                    return_attn_weights=return_attn_weights
                ),
                'drop_path1': DropPath(dpr[i]),
                'norm_post_attn': RMSNorm(embed_dim),
                'norm2': RMSNorm(embed_dim),
                'ffn': SwiGLU(embed_dim, embed_dim * 4),
                'dropout_ffn': nn.Dropout(dropout_ffn),
                'drop_path2': DropPath(dpr[i]),
                'norm_post_ffn': RMSNorm(embed_dim),
            })
            self.layers.append(layer)

        # Cross-attention for multi-scale fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_attn, batch_first=True
        )
        self.cross_norm = RMSNorm(embed_dim)

        # Confounder embeddings
        self.age_proj = nn.Linear(1, embed_dim_age)
        self.ext_embed = nn.Embedding(2, embed_dim_ext)

        # Final classifier
        total_input = embed_dim + self.patch_embed_dim + embed_dim_age + embed_dim_ext
        self.classifier = nn.Sequential(
            nn.Linear(total_input, embed_dim, bias=False),
            RMSNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            RMSNorm(embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_classifier * 0.5),
            nn.Linear(embed_dim // 2, num_classes, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        """Improved weight initialization"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization for most layers
                if any(kw in name for kw in ['out_proj', 'w2']):
                    # Smaller init for residual projections
                    nn.init.xavier_uniform_(module.weight, gain=0.02)
                elif 'classifier' in name:
                    # Even smaller for classifier
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                else:
                    # Standard Xavier for other layers
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor, age: torch.Tensor, ext: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.feature_dim, f"Expected feature_dim={self.feature_dim}, got {D}"

        # Input processing
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        # Short-scale pathway
        x_short = self.patch_embed(x)
        global_short = x_short.mean(dim=1)

        # Main transformer pathway
        if self.return_attn_weights:
            self.last_attention_maps = {'main': [], 'cross': None}
        
        for i, layer in enumerate(self.layers):
            # Pre-norm attention with residual
            residual = x
            x = layer['norm1'](x)
            x = layer['attn'](x)
            x = layer['drop_path1'](x)  # Stochastic depth
            x = residual + x
            x = layer['norm_post_attn'](x)

            # Pre-norm FFN with residual
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn'](x)
            x = layer['dropout_ffn'](x)  # FFN dropout
            x = layer['drop_path2'](x)  # Stochastic depth
            x = residual + x
            x = layer['norm_post_ffn'](x)
            
            # Collect attention weights if requested
            if self.return_attn_weights and hasattr(layer['attn'], 'last_attn_weights'):
                if layer['attn'].last_attn_weights is not None:
                    self.last_attention_maps['main'].append(
                        layer['attn'].last_attn_weights.cpu()
                    )

        # Multi-scale fusion via cross-attention
        x_short_up = F.interpolate(
            x_short.transpose(1, 2), size=T, mode='linear', align_corners=False
        ).transpose(1, 2)

        # Pad short-scale to match embed_dim
        pad_dim = self.embed_dim - x_short_up.size(-1)
        if pad_dim > 0:
            padding = torch.zeros(B, T, pad_dim, device=x.device, dtype=x.dtype)
            x_short_padded = torch.cat([x_short_up, padding], dim=-1)
        else:
            x_short_padded = x_short_up[:, :, :self.embed_dim]

        # Cross-attention
        if self.return_attn_weights:
            cross_attended, cross_attn_weights = self.cross_attn(
                x, x_short_padded, x_short_padded, need_weights=True
            )
            self.last_attention_maps['cross'] = cross_attn_weights.cpu()
        else:
            cross_attended, _ = self.cross_attn(x, x_short_padded, x_short_padded)
        
        x = self.cross_norm(x + cross_attended)

        # Temporal pooling with learned attention
        attn_weights = F.softmax(self.temporal_attn(x), dim=1)
        global_main = torch.sum(x * attn_weights, dim=1)

        # Confounder embeddings
        age_emb = self.age_proj(age.unsqueeze(1))
        ext_emb = self.ext_embed(ext)

        # Final classification
        combined = torch.cat([global_main, global_short, age_emb, ext_emb], dim=1)
        logits = self.classifier(combined).squeeze(-1)
        
        return logits

    def get_attention_maps(self) -> Optional[Dict[str, Any]]:
        """Return stored attention maps if available"""
        return self.last_attention_maps if self.return_attn_weights else None

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================
# CONFIGURATION
# ======================
def create_bbtransformer(config: dict) -> BBTransformer:
    """Factory function to create a BBTransformer with full configuration support"""
    # Default configuration based on best clinically validated trial (Composite Score: 0.7103)
    default_config = {
        # Core architecture
        'feature_dim': 414,
        'num_classes': 1,
        'embed_dim': 512,
        'num_heads': 16,
        'num_layers': 7,
        # Dropout configuration
        'dropout_input': 0.17,
        'dropout_patch': 0.15,
        'dropout_attn': 0.15,
        'dropout_ffn': 0.24,
        'dropout_classifier': 0.04,
        'dropout_temporal': 0.16,
        # Confounder embeddings
        'embed_dim_age': 32,
        'embed_dim_ext': 16,
        # Patching
        'patch_size': 3,
        'patch_embed_ratio': 0.75,
        # Temporal attention
        'temp_attn_hidden': 512,
        # GQA configuration
        'n_kv_heads': 8,
        # Regularization
        'stochastic_depth_rate': 0.10,
        # Debug/interpretability
        'return_attn_weights': False
    }
    
    # Override defaults with user-provided config (if any)
    default_config.update(config)
    
    return BBTransformer(
        feature_dim=default_config['feature_dim'],
        num_classes=default_config['num_classes'],
        embed_dim=default_config['embed_dim'],
        num_heads=default_config['num_heads'],
        num_layers=default_config['num_layers'],
        dropout_input=default_config['dropout_input'],
        dropout_patch=default_config['dropout_patch'],
        dropout_attn=default_config['dropout_attn'],
        dropout_ffn=default_config['dropout_ffn'],
        dropout_classifier=default_config['dropout_classifier'],
        dropout_temporal=default_config['dropout_temporal'],
        embed_dim_age=default_config['embed_dim_age'],
        embed_dim_ext=default_config['embed_dim_ext'],
        patch_size=default_config['patch_size'],
        patch_embed_ratio=default_config['patch_embed_ratio'],
        temp_attn_hidden=default_config['temp_attn_hidden'],
        n_kv_heads=default_config['n_kv_heads'],
        return_attn_weights=default_config['return_attn_weights'],
        stochastic_depth_rate=default_config['stochastic_depth_rate']
    )


def load_pretrained_bbtransformer(weights_path: str, config: dict) -> BBTransformer:
    """Load a pretrained BBTransformer model with flexible weight loading"""
    model = create_bbtransformer(config)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Filter out keys that don't match current model
    model_state = model.state_dict()
    pretrained_state = {
        k: v for k, v in state_dict.items() 
        if k in model_state and v.shape == model_state[k].shape
    }
    
    # Log what's being loaded
    missing_keys = set(model_state.keys()) - set(pretrained_state.keys())
    unexpected_keys = set(pretrained_state.keys()) - set(model_state.keys())
    
    if missing_keys:
        print(f"Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys in checkpoint: {unexpected_keys}")
    
    # Update model with pretrained weights
    model_state.update(pretrained_state)
    model.load_state_dict(model_state)
    
    print(f"Loaded {len(pretrained_state)}/{len(model_state)} weights from checkpoint")
    
    return model

