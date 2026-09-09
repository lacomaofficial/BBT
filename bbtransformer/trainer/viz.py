# bbtransformer\trainer\viz.py
"""Publication-grade visualization module for BBTransformer."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Colormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix as sk_cm, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

# Optional imports for brain plotting
try:
    from gt_map import GlasserTianParcellator, plot_gt_rois
    HAS_GT_MAP = True
except ImportError:
    HAS_GT_MAP = False

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & PALETTE
# ============================================================================
N_ROIS = 414
EPSILON = 1e-9
TEXT_OFFSET_RATIO = 0.01
DEFAULT_DPI = 300
FIGURE_DPI = 150

GREEN = "#009E73"
GREEN_DARK = "#006B4E"
GREEN_LIGHT = "#B2DFDB"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#999999"

GREEN_CMAP = LinearSegmentedColormap.from_list(
    "green_importance", [GREEN_LIGHT, GREEN, GREEN_DARK]
)

# Register custom colormap globally
try:
    plt.colormaps.register(GREEN_CMAP, name="green_importance")
except ValueError:
    pass  # Already registered

NETWORK_PALETTE: Dict[str, str] = {
    "Visual":           "#4477AA", "Motor":            "#EE6677",
    "Somatosensory":    "#228833", "DorsalAttention":  "#CCBB44",
    "VentralAttention": "#66CCEE", "DefaultMode":      "#AA3377",
    "Language":         "#EE8866", "Frontoparietal":   "#999933",
    "CinguloOpercular": "#BBCC33", "Limbic":           "#6699CC",
    "Auditory":         "#882255", "Temporal":         "#DDCC77",
    "Other":            "#AAAAAA", "Thalamus":         "#44AA99",
    "BasalGanglia":     "#994455",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.dpi": FIGURE_DPI, "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

# ============================================================================
# ATLAS & METADATA HELPERS
# ============================================================================
@lru_cache(maxsize=1)
def ensure_unified_atlas() -> Path:
    """Create GT_414ROIs_atlas.nii.gz from separate Glasser + Tian atlases."""
    if not HAS_GT_MAP:
        raise ImportError("gt_map is required for brain plotting but is not installed.")

    import nibabel as nib
    from nilearn.image import new_img_like

    parc = GlasserTianParcellator()
    unified_path = parc.atlas_dir / "GT_414ROIs_atlas.nii.gz"

    if unified_path.exists():
        logger.info("Unified atlas found: %s", unified_path)
        return unified_path

    logger.info("Creating unified atlas from separate Glasser + Tian files...")
    glasser_img = nib.load(str(parc.glasser_nii))
    tian_img = nib.load(str(parc.tian_nii))

    glasser_data = glasser_img.get_fdata().astype(np.int32)
    tian_data = tian_img.get_fdata().astype(np.int32)

    # Shift Tian subcortical labels to start after Glasser cortical labels
    tian_shifted = np.where(tian_data > 0, tian_data + 360, 0).astype(np.int32)
    combined = glasser_data + tian_shifted

    unified_img = new_img_like(glasser_img, combined)
    nib.save(unified_img, str(unified_path))
    logger.info("Unified atlas created: %s", unified_path)
    return unified_path


@lru_cache(maxsize=1)
def load_roi_metadata() -> pd.DataFrame:
    """Load 414-ROI metadata from gt_map's bundled atlas (cached)."""
    if not HAS_GT_MAP:
        raise ImportError("gt_map is required for metadata loading.")
    from gt_map import get_bundled_atlas_dir
    df = pd.read_csv(get_bundled_atlas_dir() / "roi_labels.csv")
    if len(df) != N_ROIS:
        raise ValueError(f"Expected {N_ROIS} ROIs in metadata, found {len(df)}.")
    return df


def _label(row: pd.Series) -> str:
    hemisphere_code = "L" if row["hemisphere"] == "Left" else "R"
    return f"{hemisphere_code} - {row['region_full_name']}"


def _top(scores: np.ndarray, n: int) -> np.ndarray:
    return np.argsort(scores)[-n:][::-1]


def _net_color(system: str) -> str:
    return NETWORK_PALETTE.get(system, "#AAAAAA")


def _save(fig: plt.Figure, path: Optional[Union[str, Path]]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    logger.info("Saved figure to %s", path)


def _get_cmap(cmap: Union[str, Colormap]) -> Colormap:
    """Safely retrieve a matplotlib colormap."""
    if isinstance(cmap, str):
        try:
            return plt.get_cmap(cmap)
        except ValueError as e:
            raise ValueError(f"Invalid colormap name '{cmap}': {e}")
    return cmap


# ============================================================================
# 1. EVALUATION DASHBOARD
# ============================================================================
def plot_results(
    metrics: Dict[str, float],
    probs: np.ndarray,
    targets: np.ndarray,
    target_name: str = "Target",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    cmap_cm: Union[str, Colormap] = "light_green",
) -> plt.Figure:
    """Generate a 2x2 evaluation dashboard: ROC, PR, Calibration, Confusion."""
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Diagnostic Performance: {target_name}",
                 fontsize=14, fontweight="bold", y=0.98)

    # ROC Curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(targets, probs)
    ax.fill_between(fpr, tpr, alpha=0.15, color=ORANGE)
    ax.plot(fpr, tpr, color=ORANGE, lw=2)
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=0.8)
    ax.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.text(0.55, 0.15, f"AUC = {metrics['roc_auc']:.3f}",
            transform=ax.transAxes, fontsize=12, fontweight="bold", color=ORANGE)

    # Precision-Recall Curve
    ax = axes[0, 1]
    prec, rec, _ = precision_recall_curve(targets, probs)
    ap = average_precision_score(targets, probs)
    ax.fill_between(rec, prec, alpha=0.15, color=BLUE)
    ax.plot(rec, prec, color=BLUE, lw=2)
    ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax.text(0.55, 0.15, f"AP = {ap:.3f}",
            transform=ax.transAxes, fontsize=12, fontweight="bold", color=BLUE)

    # Calibration Curve
    ax = axes[1, 0]
    fp, mp = calibration_curve(targets, probs, n_bins=15, strategy="uniform")
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=0.8, label="Perfect Calibration")
    ax.plot(mp, fp, "o-", color=GREEN, lw=2, ms=5, label="Model")
    brier = metrics.get('brier_score', brier_score_loss(targets, probs))
    ax.set(title=f"Calibration Curve (Brier = {brier:.3f})",
           xlabel="Mean Predicted Probability", ylabel="Fraction of Positives")
    ax.legend(loc="upper left", fontsize=9)

    # Confusion Matrix
    ax = axes[1, 1]
    cm = np.asarray(metrics["confusion_matrix"])
    pct = cm / len(targets) * 100
    annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1f}%)" for j in range(2)]
                      for i in range(2)])

    cm_cmap = sns.light_palette(GREEN, as_cmap=True) if cmap_cm == "light_green" else _get_cmap(cmap_cm)

    sns.heatmap(cm, annot=annot, fmt="", cmap=cm_cmap, ax=ax, cbar=False,
                xticklabels=["Control", "Case"], yticklabels=["Control", "Case"],
                annot_kws={"fontsize": 13, "fontweight": "bold"})
    ax.set(title="Confusion Matrix", xlabel="Predicted Label", ylabel="Actual Label")

    # Footer metrics
    f1 = metrics.get('f1', f1_score(targets, (probs > 0.5).astype(int), zero_division=0))
    acc = metrics.get('accuracy', accuracy_score(targets, (probs > 0.5).astype(int)))
    prec_score = metrics.get('precision', precision_score(targets, (probs > 0.5).astype(int), zero_division=0))
    rec_score = metrics.get('recall', recall_score(targets, (probs > 0.5).astype(int), zero_division=0))
    n_cases = int(targets.sum())

    fig.text(0.5, -0.02,
             f"F1 = {f1:.3f}  |  Acc = {acc:.3f}  |  "
             f"Prec = {prec_score:.3f}  |  Rec = {rec_score:.3f}  |  "
             f"N = {len(targets)} ({n_cases} cases)",
             ha="center", fontsize=10, color="#555", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#DDD"))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    _save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================================
# 2. IMPORTANCE BARS
# ============================================================================
def plot_importance(
    importance_scores: np.ndarray,
    top_n: int = 15,
    target_name: str = "Target",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    cmap: Union[str, Colormap] = "green_importance",
) -> plt.Figure:
    """Horizontal bar chart with green gradient (darker = more important)."""
    if importance_scores.shape != (N_ROIS,):
        raise ValueError(f"Expected shape ({N_ROIS},), got {importance_scores.shape}")

    meta = load_roi_metadata()
    idx = _top(importance_scores, top_n)
    scores = importance_scores[idx]
    rows = meta.iloc[idx]

    labels = [_label(r) for _, r in rows.iterrows()]
    norm = (scores - scores.min()) / (scores.max() - scores.min() + EPSILON)

    colormap = _get_cmap(cmap)
    colors = [colormap(v) for v in norm]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    ax.barh(range(top_n), scores, color=colors, edgecolor="white",
            linewidth=0.5, height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (Δ Loss)")
    ax.set_title(f"Top {top_n} Biomarker Regions — {target_name}",
                 fontweight="bold", pad=10)

    sm = plt.cm.ScalarMappable(cmap=colormap,
                                norm=plt.Normalize(scores.min(), scores.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Importance", fontsize=9)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================================
# 3. BRAIN OVERLAY
# ============================================================================
def plot_brain(
    importance_scores: np.ndarray,
    top_n: int = 5,
    target_name: str = "Target",
    label_type: str = "full",
    cmap: Union[str, Colormap] = "Greens",
    alpha: float = 0.85,
    show: bool = True,
) -> None:
    """Plot top N important ROIs on MNI template via gt_map."""
    if importance_scores.shape != (N_ROIS,):
        raise ValueError(f"Expected shape ({N_ROIS},), got {importance_scores.shape}")

    ensure_unified_atlas()
    idx = _top(importance_scores, top_n).tolist()
    cmap_name = cmap if isinstance(cmap, str) else cmap.name

    # plot_gt_rois does not accept display_mode; removed to prevent TypeError
    plot_gt_rois(
        indices=idx,
        title=f"Top {top_n} Predictive Regions — {target_name}",
        label_type=label_type,
        cmap=cmap_name,
        alpha=alpha,
    )
    
    if show:
        plt.show()


# ============================================================================
# 4. NETWORK SUMMARY
# ============================================================================
def plot_network_summary(
    importance_scores: np.ndarray,
    target_name: str = "Target",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    cmap: Optional[Union[str, Colormap]] = None,
) -> plt.Figure:
    """Horizontal bar chart of total importance per functional system."""
    if importance_scores.shape != (N_ROIS,):
        raise ValueError(f"Expected shape ({N_ROIS},), got {importance_scores.shape}")

    meta = load_roi_metadata()
    df = meta.copy()
    df["importance"] = importance_scores

    net_imp = (
        df.groupby("functional_system")["importance"]
        .agg(["sum", "mean", "count"])
        .sort_values("sum", ascending=True)
    )

    n_systems = len(net_imp)

    # Handle dynamic cmap vs hardcoded categorical palette
    if cmap is not None:
        colormap = _get_cmap(cmap)
        norm = plt.Normalize(vmin=0, vmax=max(1, n_systems - 1))
        colors = [colormap(norm(i)) for i in range(n_systems)]
    else:
        colors = [_net_color(s) for s in net_imp.index]

    fig, ax = plt.subplots(figsize=(10, max(5, n_systems * 0.4)))
    ax.barh(range(n_systems), net_imp["sum"].values,
            color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    max_sum = net_imp["sum"].max()
    for i, (_, row) in enumerate(net_imp.iterrows()):
        offset = max_sum * TEXT_OFFSET_RATIO
        ax.text(row["sum"] + offset, i,
                f"μ={row['mean']:.4f}  n={int(row['count'])}",
                va="center", fontsize=8, color="#555")

    ax.set_yticks(range(n_systems))
    ax.set_yticklabels(net_imp.index, fontsize=9)
    ax.set_xlabel("Total Permutation Importance")
    ax.set_title(f"Predictive Importance by Functional System — {target_name}",
                 fontweight="bold", pad=10)

    plt.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


'''
# Usage Example

def generate_toy_data(n: int = 500, prevalence: float = 0.3, seed: int = 42):
    """Generate synthetic data with injected signal ROIs across systems."""
    rng = np.random.default_rng(seed)
    targets = (rng.random(n) < prevalence).astype(np.int32)
    probs = np.clip(
        targets * rng.beta(5, 2, n) + (1 - targets) * rng.beta(2, 5, n), 0, 1
    )

    importance = rng.exponential(0.005, N_ROIS)
    signal_injections = {
        7: 0.32, 90: 0.28, 147: 0.25, 263: 0.22,
        372: 0.35, 23: 0.18, 13: 0.20, 187: 0.27,
    }
    for idx, score in signal_injections.items():
        importance[idx] = score

    preds = (probs >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1": f1_score(targets, preds, zero_division=0),
        "roc_auc": roc_auc_score(targets, probs),
        "brier_score": brier_score_loss(targets, probs),
        "confusion_matrix": sk_cm(targets, preds),
    }
    return probs, targets, importance, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    probs, targets, imp, metrics = generate_toy_data()
    out = Path("results")
    out.mkdir(exist_ok=True)

    plot_results(
        metrics=metrics, probs=probs, targets=targets,
        target_name="Toy", save_path=out / "eval.png",
    )

    plot_importance(
        importance_scores=imp, top_n=15, target_name="Toy",
        save_path=out / "importance.png",
    )

    if HAS_GT_MAP:
        plot_brain(
            importance_scores=imp, top_n=5, target_name="Toy (Purples)",
            label_type="full", cmap="Pastel2_r",
        )
    else:
        logger.warning("gt_map not installed. Skipping brain overlay plot.")

    plot_network_summary(
        importance_scores=imp, target_name="Toy",
        save_path=out / "network.png", cmap="Pastel2_r",
    )

    logger.info("Visualization pipeline complete. Output directory: %s", out)

'''
