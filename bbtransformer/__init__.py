# bbtransformer/__init__.py

# Model
from .model import BBTransformer, create_bbtransformer

# Utilities
from .utils import save_model_weights, load_model_weights, load_roi_names, load_roi_metadata

# Trainer components
from .trainer.loader import prepare_fmri_data
from .trainer.train import train_model
from .trainer.exe import run_analysis
from .trainer.eval import evaluate_model
from .trainer.rank import calculate_permutation_importance, save_top_importance_to_csv
from .trainer.tune import tune_hyperparameters, run_tuning_workflow
from .trainer.pred import Diagnostic

# Visualization (NEW)
from .trainer.viz import (
    plot_results,
    plot_importance,
    plot_brain,
    plot_network_summary,
)
