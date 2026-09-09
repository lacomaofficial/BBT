# bbtransformer\trainer\eval.py

"""Model evaluation metrics for BBTransformer."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.amp import autocast
from tqdm import tqdm

# Re-export from viz for backward compatibility
from .viz import plot_results  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "evaluate_model",
    "plot_results",
]


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
) -> Tuple[Dict[str, float], List[np.ndarray], List[np.ndarray]]:
    """Evaluate model performance on a dataset.

    Args:
        model: Trained BBTransformer model.
        data_loader: DataLoader containing (fmri, age, ext, labels) batches.

    Returns:
        Tuple of (metrics_dict, list_of_probs, list_of_targets).
        metrics_dict contains: accuracy, precision, recall, f1, roc_auc,
        brier_score, confusion_matrix.
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds: List[int] = []
    all_probs: List[float] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for fmri, age, ext, labels in tqdm(data_loader, desc="Evaluation", leave=False):
            fmri = fmri.to(device)
            age = age.to(device)
            ext = ext.to(device)

            with autocast(device_type=device.type):
                logits = model(fmri, age, ext)
                probs = torch.sigmoid(logits).cpu().numpy()

            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.ravel().tolist())
            all_preds.extend(preds.ravel().tolist())
            all_targets.extend(labels.cpu().numpy().astype(int).tolist())

    if not all_preds:
        logger.warning("No predictions generated; returning zero metrics.")
        metrics: Dict[str, float] = {
            k: 0.0
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"]
        }
        metrics["confusion_matrix"] = np.zeros((2, 2))
        return metrics, all_probs, all_targets

    probs_arr = np.array(all_probs)
    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_preds)

    metrics = {
        "accuracy": accuracy_score(targets_arr, preds_arr),
        "precision": precision_score(targets_arr, preds_arr, zero_division=0),
        "recall": recall_score(targets_arr, preds_arr, zero_division=0),
        "f1": f1_score(targets_arr, preds_arr, zero_division=0),
        "roc_auc": roc_auc_score(targets_arr, probs_arr),
        "brier_score": brier_score_loss(targets_arr, probs_arr),
        "confusion_matrix": confusion_matrix(targets_arr, preds_arr),
    }

    logger.info(
        "Evaluation complete: F1=%.4f | AUC=%.4f | Acc=%.4f | Brier=%.4f",
        metrics["f1"],
        metrics["roc_auc"],
        metrics["accuracy"],
        metrics["brier_score"],
    )

    return metrics, all_probs, all_targets
