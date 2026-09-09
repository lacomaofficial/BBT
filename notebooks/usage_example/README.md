# BBTransformer: In-Depth Fine-Tuning Tutorial

This guide provides a comprehensive walkthrough for fine-tuning a pretrained BBTransformer model on custom clinical cohorts using the refactored `run_analysis` pipeline. This version supports absolute weight paths, configurable batch sizes, decoupled project naming, and a complete 4-panel visualization suite including brain overlays.

## 1. The Complete Fine-Tuning Script

Save this as `run_analysis.py`. This script demonstrates robust path handling and all new configuration options.

```python
import logging
from pathlib import Path

from bbtransformer.trainer.exe import run_analysis

# Enable detailed progress logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# --- PATH DEFINITION ---
# Use absolute paths directly. Do NOT concatenate with Path / operator.
DATA_PATH = "/mnt/data/fmri/chrt/fmri_ASD_vs_ADHD.npz"
PHENO_PATH = "/mnt/data/fmri/chrt/pheno_ASD_vs_ADHD.csv"

# Pretrained weights: absolute paths are now natively supported
PRETRAINED_WEIGHTS = "/mnt/models/weights/bbtransformer_base.pth"

# --- RUN THE FULL PIPELINE ---
results = run_analysis(
    # 1. DATA IDENTIFICATION
    target_column="target_label",           # Binary column (0/1) in CSV
    project_name="ASD_vs_ADHD_finetune",    # NEW: Decouples output naming from target column
    data_path=DATA_PATH,                    # Absolute path supported natively
    pheno_path=PHENO_PATH,                  # Absolute path supported natively
    base_dir=None,                          # Set to None when using explicit paths

    # 2. PRETRAINED WEIGHTS (TRANSFER LEARNING)
    project_root=".",                       # Base dir for ./weights/ and ./results/
    use_pretrained=True,
    pretrained_weight_file=PRETRAINED_WEIGHTS,  # Absolute OR relative to weights_dir

    # 3. FINE-TUNING HYPERPARAMETERS
    training_config={
        "epochs": 500,                      # Reduced from default 5000 for fine-tuning
        "lr": 1e-5,                         # Lower LR prevents catastrophic forgetting
        "weight_decay": 1.14e-06,           # L2 regularization strength
        "patience": 50,                     # Early stopping window
    },
    early_stop_metric="loss",               # "loss" recommended for FT; "f1" for from-scratch
    use_focal_loss=False,                   # Set True if class imbalance > 70/30

    # 4. BATCH SIZE (NEW)
    batch_size=64,                          # Configurable; was previously hardcoded

    # 5. BIOMARKER DISCOVERY (PERMUTATION IMPORTANCE)
    compute_importance=True,                # Rank all 414 brain regions
    importance_metric="loss",               # "loss" is more stable than "f1" for ranking
    importance_n_repeats=30,                # 30=research-grade; 5=quick sanity check

    # 6. OUTPUT & REPRODUCIBILITY
    save_plots=True,                        # Generates ALL 4 plot types + brain overlay
    save_json=True,                         # Metrics + top ROIs summary
    random_seed=42,                         # Locks PyTorch/NumPy/CUDA seeds
    device=None,                            # Auto-detects CUDA; set "cpu" to force
)

logging.info(
    "Pipeline complete | F1: %.4f | AUC: %.4f | Project: %s",
    results["metrics"]["f1"],
    results["metrics"]["roc_auc"],
    results["project_name"],
)
```

## 2. Parameter Deep Dive

### New & Updated Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `project_name` | `str \| None` | `target_column` | Name for all output files and plot titles. Decouples naming from clinical label. |
| `batch_size` | `int` | `64` | Batch size for all data loaders. Reduce if OOM during importance computation. |
| `pretrained_weight_file` | `str` | `None` | **Now supports absolute paths.** Relative paths resolve against `project_root/weights/`. |
| `device` | `str \| None` | `None` | `"cuda"`, `"cpu"`, or `None` for auto-detection. |

### Weight Path Resolution (Fixed)

The previous double-nesting bug has been resolved. Both patterns now work correctly:

| Scenario | What to Pass | Internal Resolution |
| :--- | :--- | :--- |
| Weights in `project_root/weights/` | `"neuroX.pth"` | `{project_root}/weights/neuroX.pth` ✅ |
| Weights elsewhere (absolute) | `"/mnt/data/neuroX.pth"` | `/mnt/data/neuroX.pth` ✅ |
| Relative subdirectory | `"experiments/run1/model.pth"` | `{project_root}/weights/experiments/run1/model.pth` ✅ |

> **Note:** `save_model_weights` now saves explicitly to `{project_root}/weights/weights_{project_name}.pth`, respecting your `project_root` setting instead of a hardcoded global directory.

### Training Configuration for Fine-Tuning vs. From-Scratch

| Parameter | Fine-Tuning | From Scratch | Rationale |
| :--- | :--- | :--- | :--- |
| `epochs` | 300–800 | 5000 | Pretrained weights need fewer cycles to adapt |
| `lr` | 1e-5 – 5e-6 | 2.3e-5 | Lower LR preserves pretrained representations |
| `patience` | 30–50 | 90 | Convergence is faster; stop earlier to avoid overfitting |
| `early_stop_metric` | `"loss"` | `"f1"` | Loss is smoother during FT; F1 can be noisy early on |
| `use_focal_loss` | Only if imbalanced | Default False | Adaptive focal loss helps when prevalence <30% or >70% |
| `batch_size` | 32–64 | 64 | Reduce if GPU memory is limited during importance passes |

### Permutation Importance Tradeoffs

`importance_n_repeats` controls the accuracy/speed tradeoff:

-   **2–5 repeats:** ~10–20 minutes. Debugging and quick checks only. High variance.
-   **30 repeats:** ~2 hours. Research-grade stability. Recommended for publication.
-   **Metric choice:** `"loss"` measures continuous degradation and is generally more stable than `"f1"`, which can be noisy due to threshold effects at 0.5.

## 3. Output Files Explained

With `save_plots=True` and `compute_importance=True`, the pipeline generates **all** visualization artifacts:

| File | Source Function | Contents |
| :--- | :--- | :--- |
| `{project}_results.png` | `plot_results` | 2×2 dashboard: ROC, PR curve, Calibration, Confusion Matrix |
| `importance_{project}_plot.png` | `plot_importance` | Horizontal bar chart of top 5 ROIs with green gradient |
| `network_{project}.png` | `plot_network_summary` | Total importance per functional system (no legend; y-axis labels) |
| *(interactive window)* | `plot_brain` | MNI brain overlay of top 5 predictive regions via `gt_map` |
| `importance_{project}.csv` | `save_top_importance_to_csv` | Top 5 regions with full anatomical metadata |
| `{project}_results.json` | JSON export | All metrics + top 10 ROIs + dataset metadata |
| `weights/weights_{project}.pth` | `save_model_weights` | Fine-tuned model state dict (safe format, `weights_only=True`) |

> **Brain Overlay Note:** `plot_brain` requires `gt_map` and automatically generates a unified Glasser+Tian atlas on first run. If `gt_map` is unavailable, the pipeline logs a warning and continues without crashing. All other plots are unaffected.

## 4. Troubleshooting Common Failures

| Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `ValueError: Missing columns in phenotype` | CSV missing `Age`, `Sex`, or target column | Verify exact column names (case-sensitive) |
| `FileNotFoundError: Pretrained weights` | Incorrect absolute path or missing file | Verify file exists; absolute paths are used as-is |
| `RuntimeError: Failed to load pretrained weights` | Architecture mismatch between weights and model | Ensure `model_config` matches pretraining architecture |
| Model predicts only one class | Severe class imbalance without focal loss | Set `use_focal_loss=True`; verify label distribution |
| Brain overlay not displayed | `gt_map` not installed or atlas generation failed | Install `gt_map`; check warning log for details |
| OOM during importance computation | Batch size too large for permutation passes | Reduce `batch_size` parameter (e.g., 32 or 16) |
| Outputs saved to wrong directory | Using old code before `exe.py` refactor | Update to latest `exe.py` with explicit `save_path` in `save_model_weights` |
| Importance plot is noisy/unstable | Too few permutation repeats | Increase `importance_n_repeats` to ≥20 |

## Sources

-   `bbtransformer/trainer/exe.py` – `run_analysis` with absolute path support, `project_name`, `batch_size`, and full 4-plot `_save_results`
-   `bbtransformer/trainer/viz.py` – `plot_results`, `plot_importance`, `plot_brain`, `plot_network_summary`
-   `bbtransformer/trainer/eval.py` – `evaluate_model` (metrics computation only; plotting moved to `viz.py`)
-   `bbtransformer/trainer/rank.py` – `calculate_permutation_importance`, `save_top_importance_to_csv` (plotting moved to `viz.py`)
-   `bbtransformer/trainer/loader.py` – `prepare_fmri_data` with stratified splits and age normalization
-   `bbtransformer/trainer/train.py` – Ranger21 optimizer, AdaptiveFocalLoss, early stopping
-   `bbtransformer/utils.py` – `load_model_weights`, `save_model_weights`, `load_roi_metadata`
