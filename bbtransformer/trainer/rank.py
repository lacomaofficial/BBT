# bbtransformer\trainer\rank.py
"""Permutation importance ranking for BBTransformer."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.amp import autocast
from tqdm import tqdm

from .eval import evaluate_model
from .viz import plot_importance  # Import from new viz module
from ..utils import load_roi_names

logger = logging.getLogger(__name__)

# Re-export for backward compatibility so existing code doesn't break
__all__ = [
    "calculate_permutation_importance",
    "save_top_importance_to_csv",
    "plot_importance",
]


def calculate_permutation_importance(
    model: torch.nn.Module,
    data_loader,
    feature_dim: int,
    n_repeats: int = 10,
    seed: int = 42,
    target_name: str = "target",
    metric: str = "f1",
) -> np.ndarray:
    """Calculate permutation importance for brain regions.

    Args:
        model: Trained BBTransformer.
        data_loader: DataLoader with validation data.
        feature_dim: Number of brain regions (e.g., 414).
        n_repeats: Number of permutation repeats per region.
        seed: Random seed for reproducibility.
        target_name: Name of target variable (used in logging).
        metric: Importance metric ("f1" or "loss").

    Returns:
        Array of importance scores with shape (feature_dim,).
    """
    if metric not in ("f1", "loss"):
        raise ValueError(f"metric must be 'f1' or 'loss', got '{metric}'")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = next(model.parameters()).device

    if metric == "f1":
        baseline_metrics, _, _ = evaluate_model(model, data_loader)
        baseline_score = baseline_metrics["f1"]
    else:
        baseline_score = _compute_validation_loss(model, data_loader, device)

    importance_scores = np.zeros(feature_dim)
    logger.info(
        "Calculating permutation importance (%s) for %d brain regions...",
        metric, feature_dim,
    )

    for region_idx in tqdm(range(feature_dim), desc="Permuting regions"):
        region_scores = []
        for _ in range(n_repeats):
            if metric == "f1":
                score = _permute_region_f1(model, data_loader, device, region_idx)
            else:
                score = _permute_region_loss(model, data_loader, device, region_idx)
            region_scores.append(score)

        if metric == "f1":
            importance_scores[region_idx] = baseline_score - np.mean(region_scores)
        else:
            # Higher loss delta = more important
            importance_scores[region_idx] = np.mean(region_scores) - baseline_score

    return importance_scores


def _permute_region_f1(model, data_loader, device, region_idx: int) -> float:
    """Compute F1 after permuting a single region across the batch."""
    all_preds, all_probs, all_targets = [], [], []
    for fmri, age, ext, labels in data_loader:
        fmri = fmri.to(device)
        age = age.to(device)
        ext = ext.to(device)

        perm = torch.randperm(fmri.size(0), device=device)
        fmri_perm = fmri.clone()
        fmri_perm[:, :, region_idx] = fmri[perm, :, region_idx]

        with torch.no_grad(), autocast(device_type=device.type):
            logits = model(fmri_perm, age, ext)
            probs = torch.sigmoid(logits).cpu().numpy()

        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy().astype(int))

    return f1_score(all_targets, all_preds, zero_division=0)


def _permute_region_loss(model, data_loader, device, region_idx: int) -> float:
    """Compute mean BCE loss after permuting a single region across the batch."""
    total_loss = 0.0
    n_samples = 0
    for fmri, age, ext, labels in data_loader:
        fmri = fmri.to(device)
        age = age.to(device)
        ext = ext.to(device)
        labels = labels.to(device).float()

        perm = torch.randperm(fmri.size(0), device=device)
        fmri_perm = fmri.clone()
        fmri_perm[:, :, region_idx] = fmri[perm, :, region_idx]

        with torch.no_grad(), autocast(device_type=device.type):
            logits = model(fmri_perm, age, ext)
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
            total_loss += loss.item()
            n_samples += labels.size(0)

    return total_loss / n_samples if n_samples > 0 else float("inf")


def _compute_validation_loss(model, data_loader, device) -> float:
    """Compute average BCE loss on validation set without permutation."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for fmri, age, ext, labels in data_loader:
            fmri = fmri.to(device)
            age = age.to(device)
            ext = ext.to(device)
            labels = labels.to(device).float()

            with autocast(device_type=device.type):
                logits = model(fmri, age, ext)
                loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
                total_loss += loss.item()
                n_samples += labels.size(0)

    return total_loss / n_samples if n_samples > 0 else float("inf")


def save_top_importance_to_csv(
    importance_scores: np.ndarray,
    roi_metadata: Optional[pd.DataFrame] = None,
    target_name: str = "target",
    top_n: int = 30,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Save top permutation importance scores to CSV with full ROI metadata.

    Args:
        importance_scores: Array of importance scores (length = 414).
        roi_metadata: Full ROI metadata DataFrame from load_roi_metadata().
        target_name: Name of target variable.
        top_n: Number of top regions to save.
        save_path: Output CSV path.

    Returns:
        Full importance table (all 414 regions, sorted descending).
    """
    assert len(importance_scores) == 414, f"Expected 414 regions, got {len(importance_scores)}"

    col_name = f"importance_{target_name}"

    if roi_metadata is not None:
        df = roi_metadata.copy()
        df[col_name] = importance_scores
    else:
        roi_names = load_roi_names()
        df = pd.DataFrame({
            "roi_index": np.arange(414),
            "roi_name": roi_names,
            col_name: importance_scores,
        })

    df = df.sort_values(col_name, ascending=False).reset_index(drop=True)

    if save_path is None:
        save_path = f"importance_{target_name}.csv"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(top_n).to_csv(save_path, index=False)
    logger.info("Saved top %d features to: %s", top_n, save_path)

    return df
