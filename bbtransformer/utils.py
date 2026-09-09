# utils.py

# Import Essentials
import os
import time
import torch
import pandas as pd
from typing import Optional, List
import importlib.resources as pkg_resources

# ======================
# MODEL WEIGHT UTILITIES
# ======================
WEIGHTS_DIR = 'weights'
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def save_model_weights(
    model,
    target_name=None,
    save_path=None,
    include_metadata=True,
    metadata=None,
    safe_format=True  # ← NEW: save tensors-only if True
):
    """
    Save model weights. If safe_format=True, saves only tensors (compatible with weights_only=True).
    """
    if save_path is None:
        filename = f"weights_{target_name}.pth" if target_name else "new_weights.pth"
        save_path = os.path.join(WEIGHTS_DIR, filename)
    elif not os.path.isabs(save_path) and not save_path.startswith(WEIGHTS_DIR):
        save_path = os.path.join(WEIGHTS_DIR, save_path)

    try:
        if safe_format:
            # ✅ SAFE: Only save state dict (no pickled objects)
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved SAFE weights (weights_only=True compatible) to: {save_path}")
        else:
            # Legacy format with metadata
            save_dict = {'model_state_dict': model.state_dict()}
            if include_metadata:
                save_dict.update({
                    'timestamp': time.time(),
                    'target': target_name,
                    'metadata': metadata
                })
            torch.save(save_dict, save_path)
            print(f"✓ Saved FULL checkpoint (with metadata) to: {save_path}")
        return True
    except Exception as e:
        print(f"! Error saving weights: {e}")
        return False

def load_model_weights(model, device, weight_paths):
    """Load pretrained weights safely (CPU-first, shape-checked, no OOM)"""
    model_state_dict = model.state_dict()
    for weight_path in weight_paths:
        # Resolve full path
        full_path = weight_path if os.path.isabs(weight_path) else os.path.join(WEIGHTS_DIR, weight_path)
        
        if not os.path.exists(full_path):
            print(f"! File not found: {full_path}")
            continue

        try:
            print(f"Attempting SAFE load (on CPU first): {full_path}")
            # 🔥 CRITICAL: Load to CPU first to avoid OOM
            state_dict = torch.load(full_path, map_location='cpu', weights_only=True)
            
            # Filter by key and shape
            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            
            if not pretrained_dict:
                print("⚠️ No matching keys/shapes found in checkpoint.")
                continue
                
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"✓ Successfully loaded {len(pretrained_dict)} layers from: {full_path}")
            return True
            
        except Exception as e1:
            print(f"! SAFE load failed: {e1}")
            try:
                print(f"Attempting FULL load (on CPU first): {full_path}")
                state_dict = torch.load(full_path, map_location='cpu', weights_only=False)
                raw_dict = state_dict.get('model_state_dict', state_dict)
                
                pretrained_dict = {
                    k: v for k, v in raw_dict.items()
                    if k in model_state_dict and v.shape == model_state_dict[k].shape
                }
                
                if not pretrained_dict:
                    print("⚠️ No matching keys/shapes in FULL checkpoint.")
                    continue
                    
                model.load_state_dict(pretrained_dict, strict=False)
                print(f"✓ Successfully loaded {len(pretrained_dict)} layers from: {full_path}")
                return True
                
            except Exception as e2:
                print(f"! FULL load also failed: {e2}")
                continue
                
    print("! No valid weights found — will train from scratch.")
    return False

def list_available_weights():
    """List all available weight files"""
    files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.pth')]
    files.sort()
    return files if files else ['Train from Scratch']


# ======================
# GT ATLAS UTILITIES
# ======================


def load_roi_metadata():
    """Load full 414 ROI metadata from Glasser+Tian atlas."""
    with pkg_resources.open_text('bbtransformer', 'roi_labels.csv') as f:
        df = pd.read_csv(f)
    assert len(df) == 414, f"Expected 414 ROIs, got {len(df)}"
    return df  # Returns full DataFrame with all columns

def load_roi_names():
    """Backward-compatible helper: return only ROI names."""
    return load_roi_metadata()['roi_name'].values
