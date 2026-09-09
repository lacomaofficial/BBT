# bbtransformer\trainer\exe.py
"""BBTransformer analysis pipeline execution module."""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from ..model import create_bbtransformer
from ..utils import load_model_weights, load_roi_metadata, save_model_weights
from .eval import evaluate_model
from .loader import prepare_fmri_data
from .rank import calculate_permutation_importance, save_top_importance_to_csv
from .train import train_model
from .viz import plot_brain, plot_importance, plot_network_summary, plot_results

logger = logging.getLogger(__name__)

DEFAULT_MODEL_HP: Dict[str, Any] = {
    "embed_dim": 512,
    "num_heads": 16,
    "num_layers": 7,
    "n_kv_heads": 4,
    "dropout_input": 0.18,
    "dropout_patch": 0.16,
    "dropout_attn": 0.15,
    "dropout_ffn": 0.25,
    "dropout_classifier": 0.07,
    "dropout_temporal": 0.16,
    "embed_dim_age": 32,
    "embed_dim_ext": 16,
    "patch_size": 3,
    "patch_embed_ratio": 0.75,
    "temp_attn_hidden": 512,
    "return_attn_weights": False,
    "stochastic_depth_rate": 0.07,
}

DEFAULT_TRAIN_CFG: Dict[str, Any] = {
    "epochs": 5000,
    "lr": 2.3157e-05,
    "weight_decay": 1.14e-06,
    "patience": 90,
}


def _resolve_paths(
    data_path: Optional[str],
    pheno_path: Optional[str],
    base_dir: Optional[str],
    target_column: str,
) -> tuple[str, str]:
    """Resolve fMRI and phenotype file paths to absolute string paths."""
    if data_path is not None and pheno_path is not None:
        return str(Path(data_path).resolve()), str(Path(pheno_path).resolve())
    if base_dir is not None:
        resolved_data = str(Path(base_dir).resolve() / f"fmri_{target_column}.npz")
        resolved_pheno = str(Path(base_dir).resolve() / f"pheno_{target_column}.csv")
        return resolved_data, resolved_pheno
    raise ValueError("Either (data_path and pheno_path) or base_dir must be provided.")


def _validate_inputs(
    data_path: str,
    pheno_path: str,
    weight_path: Optional[str],
    pheno_df: pd.DataFrame,
    target_column: str,
) -> None:
    """Validate that all required input files and columns exist."""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"fMRI data not found: {data_path}")
    if not Path(pheno_path).exists():
        raise FileNotFoundError(f"Phenotype file not found: {pheno_path}")
    if weight_path is not None and not Path(weight_path).exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weight_path}")

    required_columns = [target_column, "Age", "Sex"]
    missing = [col for col in required_columns if col not in pheno_df.columns]
    if missing:
        raise ValueError(f"Missing columns in phenotype: {missing}")


def _configure_seeds(random_seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def _build_model_config(
    metadata: Dict[str, Any], model_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge user model config with defaults and dataset-derived feature dim."""
    config = DEFAULT_MODEL_HP.copy()
    config["feature_dim"] = metadata["feature_dim"]
    config["num_classes"] = 1
    if model_config is not None:
        config.update(model_config)
    return config


def _build_training_config(training_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge user training config with defaults."""
    config = DEFAULT_TRAIN_CFG.copy()
    if training_config is not None:
        config.update(training_config)
    return config


def _save_results(
    project_name: str,
    metrics: Dict[str, Any],
    metadata: Dict[str, Any],
    importance_scores: Optional[np.ndarray],
    roi_metadata: pd.DataFrame,
    output_dir: Path,
    save_plots_flag: bool,
    save_json_flag: bool,
    probs: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Save all evaluation plots, importance CSVs, brain overlays, and JSON summary."""
    # 1. Evaluation Dashboard (always saved when flag is set)
    if save_plots_flag:
        plot_results(
            metrics=metrics,
            probs=probs,
            targets=targets,
            target_name=project_name,
            save_path=output_dir / f"{project_name}_results.png",
            show=True,
        )

    # 2-4. Importance-based visualizations (only when scores are available)
    if importance_scores is not None:
        save_top_importance_to_csv(
            importance_scores=importance_scores,
            roi_metadata=roi_metadata,
            target_name=project_name,
            top_n=5,
            save_path=output_dir / f"importance_{project_name}.csv",
        )

        if save_plots_flag:
            # 2. Top ROI Bar Chart
            plot_importance(
                importance_scores=importance_scores,
                top_n=5,
                target_name=project_name,
                save_path=output_dir / f"importance_{project_name}_plot.png",
                show=True,
            )

            # 3. Brain Overlay (graceful degradation if gt_map unavailable)
            try:
                plot_brain(
                    importance_scores=importance_scores,
                    top_n=5,
                    target_name=project_name,
                    label_type="full",
                    cmap="Pastel2_r",
                    show=True,
                )
            except Exception as e:
                logger.warning(
                    "Brain overlay skipped (gt_map may be unavailable): %s", e
                )

            # 4. Network Summary
            plot_network_summary(
                importance_scores=importance_scores,
                target_name=project_name,
                save_path=output_dir / f"network_{project_name}.png",
                show=True,
                cmap="Pastel2_r",
            )

    # 5. JSON Summary
    if save_json_flag:
        json_metrics = {
            k: float(v) if isinstance(v, (np.number, float, int)) else v
            for k, v in metrics.items()
            if k != "confusion_matrix"
        }
        json_metadata = {
            k: float(v) if isinstance(v, (np.number, float)) else v
            for k, v in metadata.items()
        }

        json_results: Dict[str, Any] = {
            "project": project_name,
            "metrics": json_metrics,
            "metadata": json_metadata,
            "importance_top_rois": None,
        }

        if importance_scores is not None:
            top_indices = np.argsort(importance_scores)[-10:][::-1]
            top_rois = []
            for idx in top_indices:
                row = roi_metadata.iloc[idx].to_dict()
                row["importance_score"] = float(importance_scores[idx])
                top_rois.append(row)
            json_results["importance_top_rois"] = top_rois

        json_path = output_dir / f"{project_name}_results.json"
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=4)
        logger.info("Results saved to JSON: %s", json_path)


def run_analysis(
    target_column: str,
    project_name: Optional[str] = None,
    data_path: Optional[str] = None,
    pheno_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    project_root: str = ".",
    pretrained_weight_file: Optional[str] = None,
    use_pretrained: bool = True,
    compute_importance: bool = True,
    random_seed: int = 42,
    device: Optional[str] = None,
    batch_size: int = 64,
    save_plots: bool = True,
    save_json: bool = True,
    early_stop_metric: str = "f1",
    use_focal_loss: bool = False,
    training_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    importance_metric: str = "loss",
    importance_n_repeats: int = 30,
) -> Dict[str, Any]:
    """Run the full BBTransformer classification pipeline.

    Args:
        target_column: Binary target column name in the phenotype file.
        project_name: Name used for all output files and plot titles.
            Defaults to target_column if not provided.
        data_path: Full path to fMRI .npz file.
        pheno_path: Full path to phenotype .csv file.
        base_dir: Legacy fallback directory for automatic path resolution.
        project_root: Root directory for weights and results.
        pretrained_weight_file: Path to pretrained weights (absolute or relative to weights_dir).
        use_pretrained: Whether to load pretrained weights.
        compute_importance: Whether to compute permutation importance.
        random_seed: Random seed for reproducibility.
        device: Device for model execution ('cpu' or 'cuda').
        batch_size: Batch size for data loaders.
        save_plots: Whether to save and display all evaluation and importance plots.
        save_json: Whether to save a JSON summary of results.
        early_stop_metric: Metric for early stopping ('f1' or 'loss').
        use_focal_loss: Whether to use adaptive focal loss.
        training_config: Override default training hyperparameters.
        model_config: Override default model hyperparameters.
        importance_metric: Metric for permutation importance ('f1' or 'loss').
        importance_n_repeats: Number of repeats for permutation importance.

    Returns:
        Dictionary containing metrics, trained model, metadata, and importance scores.
    """
    if project_name is None:
        project_name = target_column

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    root_path = Path(project_root).resolve()
    weights_dir = root_path / "weights"
    output_dir = root_path / "results"

    weights_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_seeds(random_seed)

    resolved_data_path, resolved_pheno_path = _resolve_paths(
        data_path, pheno_path, base_dir, target_column
    )

    pheno_df = pd.read_csv(resolved_pheno_path)
    features = np.load(resolved_data_path)
    fmri_data = features["data"]
    subject_ids = features["subject_ids"]

    # Support both absolute paths and filenames relative to weights_dir
    weight_path: Optional[str] = None
    if use_pretrained:
        if pretrained_weight_file is None:
            raise ValueError(
                "pretrained_weight_file must be provided when use_pretrained=True"
            )
        p = Path(pretrained_weight_file)
        weight_path = str(p.resolve()) if p.is_absolute() else str(weights_dir / p)

    _validate_inputs(
        resolved_data_path, resolved_pheno_path, weight_path, pheno_df, target_column
    )

    roi_metadata = load_roi_metadata()

    logger.info("Loading data for project='%s' (target='%s')", project_name, target_column)
    logger.info("fMRI shape: %s, Subjects: %d", fmri_data.shape, len(subject_ids))

    train_loader, val_loader, test_loader, metadata = prepare_fmri_data(
        data_path=resolved_data_path,
        pheno_path=resolved_pheno_path,
        target_column=target_column,
        age_column="Age",
        ext_column="Sex",
        batch_size=batch_size,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_seed=random_seed,
    )

    logger.info(
        "Dataset metadata: %s",
        {k: v for k, v in metadata.items() if k not in ["age_mean", "age_std"]},
    )

    torch.cuda.empty_cache()
    gc.collect()

    model_hp = _build_model_config(metadata, model_config)
    model = create_bbtransformer(model_hp)
    model.to(device)
    logger.info("Model created on %s with %d parameters", device, model.count_parameters())

    if use_pretrained and weight_path is not None:
        success = load_model_weights(model, device=device, weight_paths=[weight_path])
        if not success:
            raise RuntimeError("Failed to load pretrained weights.")
        logger.info("Pretrained weights loaded from %s", weight_path)
    else:
        logger.info("Training from scratch (no pretrained weights).")

    train_cfg = _build_training_config(training_config)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg["epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        patience=train_cfg["patience"],
        use_focal_loss=use_focal_loss,
        early_stop_metric=early_stop_metric,
    )

    # Save weights explicitly to project_root/weights, not global WEIGHTS_DIR
    final_weight_path = str(weights_dir / f"weights_{project_name}.pth")
    save_model_weights(
        trained_model,
        target_name=project_name,
        save_path=final_weight_path,
        safe_format=True,
    )

    metrics, probs, targets = evaluate_model(trained_model, test_loader)

    logger.info("Performance Metrics:")
    logger.info("  Accuracy:  %.4f", metrics["accuracy"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall:    %.4f", metrics["recall"])
    logger.info("  F1 Score:  %.4f", metrics["f1"])
    logger.info("  ROC-AUC:   %.4f", metrics["roc_auc"])

    importance_scores: Optional[np.ndarray] = None
    if compute_importance:
        logger.info(
            "Computing permutation importance (metric=%s, repeats=%d)",
            importance_metric,
            importance_n_repeats,
        )
        importance_scores = calculate_permutation_importance(
            trained_model,
            val_loader,
            feature_dim=metadata["feature_dim"],
            n_repeats=importance_n_repeats,
            seed=random_seed,
            target_name=project_name,
            metric=importance_metric,
        )
        logger.info("Permutation importance computation complete.")

    _save_results(
        project_name=project_name,
        metrics=metrics,
        metadata=metadata,
        importance_scores=importance_scores,
        roi_metadata=roi_metadata,
        output_dir=output_dir,
        save_plots_flag=save_plots,
        save_json_flag=save_json,
        probs=probs,
        targets=targets,
    )

    logger.info("Pipeline complete for project: %s", project_name)

    return {
        "metrics": metrics,
        "model": trained_model,
        "metadata": metadata,
        "importance_scores": importance_scores,
        "project_name": project_name,
        "target_column": target_column,
    }
