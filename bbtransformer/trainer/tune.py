# bbtransformer/trainer/tune.py

# Import Essentials
import os
import json
import torch
import optuna
import traceback
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List


from ..model import create_bbtransformer
from .train import train_model  
from .eval import evaluate_model
from .loader import prepare_fmri_data


# ======================
# HYPERPARAMETER TUNING 
# ======================

def optuna_objective(trial, train_loader, val_loader, feature_dim, search_config=None):
    """
    Objective function for Optuna with support for new BBTransformer features.
    Uses PyTorch's automatic FlashAttention optimization.
    """
    if search_config is None:
        search_config = {}

    # Helper to safely get config with defaults
    def get(key, default):
        return search_config.get(key, default)

    # --- Hyperparameter suggestions (enhanced for 2026 SOTA) ---
    embed_dim = trial.suggest_categorical('embed_dim', get('embed_dim', [512]))
    num_heads = trial.suggest_categorical('num_heads', get('num_heads', [12]))
    
    # CRITICAL: Validate divisibility BEFORE model creation
    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned("Invalid: embed_dim not divisible by num_heads")
    
    # FIXED: Respect user-provided n_kv_heads_options properly
    n_kv_options = get('n_kv_heads_options', None)
    if n_kv_options is not None and len(n_kv_options) > 0:
        # User specified n_kv_heads options - use them
        valid_kv_heads = [h for h in n_kv_options if num_heads % h == 0]
        if not valid_kv_heads:
            raise optuna.TrialPruned(f"No valid n_kv_heads in {n_kv_options} for num_heads={num_heads}")
        n_kv_heads = trial.suggest_categorical('n_kv_heads', valid_kv_heads)
    else:
        # Fallback to default logic
        valid_kv_heads = [h for h in [2, 4, 8] if num_heads % h == 0 and h <= num_heads]
        if not valid_kv_heads:
            valid_kv_heads = [max(1, num_heads // 4)]
        n_kv_heads = trial.suggest_categorical('n_kv_heads', valid_kv_heads)

    config = {
        'feature_dim': feature_dim,
        'num_classes': 1,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': trial.suggest_int('num_layers', *get('num_layers_range', (6, 6))),
        
        # Decoupled dropout rates - USE CUSTOM RANGES FROM search_config
        'dropout_input': trial.suggest_float('dropout_input', 
            *get('dropout_input_range', (0.16, 0.17))),
        'dropout_patch': trial.suggest_float('dropout_patch', 
            *get('dropout_patch_range', (0.17, 0.18))),
        'dropout_attn': trial.suggest_float('dropout_attn', 
            *get('dropout_attn_range', (0.15, 0.16))),
        'dropout_ffn': trial.suggest_float('dropout_ffn', 
            *get('dropout_ffn_range', (0.15, 0.17))),
        'dropout_classifier': trial.suggest_float('dropout_classifier', 
            *get('dropout_classifier_range', (0.09, 0.10))),
        'dropout_temporal': trial.suggest_float('dropout_temporal', 
            *get('dropout_temporal_range', (0.17, 0.18))),
        
        # Embedding dimensions
        'embed_dim_age': trial.suggest_categorical('embed_dim_age', get('embed_dim_age', [32])),
        'embed_dim_ext': trial.suggest_categorical('embed_dim_ext', get('embed_dim_ext', [16])),
        
        # Patching configuration
        'patch_size': trial.suggest_categorical('patch_size', get('patch_size', [3])),
        'patch_embed_ratio': trial.suggest_categorical('patch_embed_ratio', get('patch_embed_ratio', [0.75])),
        
        # Temporal attention
        'temp_attn_hidden': trial.suggest_categorical('temp_attn_hidden', get('temp_attn_hidden', [512])),
        
        # GQA configuration (must divide num_heads evenly)
        'n_kv_heads': n_kv_heads,
        
        # Stochastic depth
        'stochastic_depth_rate': trial.suggest_float('stochastic_depth_rate', 
            *get('stochastic_depth_rate_range', (0.095, 0.108))),
        
        # Interpretability
        'return_attn_weights': False,
    }

    # Optimizer hyperparameters
    lr = trial.suggest_float('lr', *get('lr_range', (2.8e-5, 3.0e-5)), log=True)
    weight_decay = trial.suggest_float('weight_decay', *get('weight_decay_range', (1.7e-6, 1.85e-6)), log=True)

    # DEBUG: Print what was actually selected
    print(f"\n[Trial {trial.number}] Selected hyperparameters:")
    print(f"  Architecture: embed_dim={embed_dim}, num_heads={num_heads}, n_kv_heads={n_kv_heads}")
    print(f"  Dropout classifier: {config['dropout_classifier']:.4f}")
    print(f"  Dropout FFN: {config['dropout_ffn']:.4f}")
    print(f"  Config n_kv_heads_options: {get('n_kv_heads_options', 'NOT SET')}")
    print(f"  Valid KV heads: {valid_kv_heads}")

    # --- Reproducibility ---
    seed = get('base_seed', 42) + trial.number
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- Model Creation ---
    print(f"\n[Trial {trial.number}] Creating model...")
    try:
        model = create_bbtransformer(config)
        n_params = model.count_parameters()
        trial.set_user_attr("n_parameters", n_params)
        print(f"[Trial {trial.number}] Model created: {n_params:,} parameters")
        
    except Exception as e:
        print(f"❌ Trial {trial.number} failed during model creation: {str(e)}")
        raise optuna.TrialPruned()

    # --- Training ---
    print(f"[Trial {trial.number}] Starting training (max {get('epochs', 5000)} epochs)...")
    try:
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=get('epochs', 5000),
            lr=lr,
            weight_decay=weight_decay,
            patience=get('patience', 100),
            use_focal_loss=get('use_focal_loss', True),  # Enable focal loss
            early_stop_metric=get('early_stop_metric', "f1")
        )
        print(f"[Trial {trial.number}] Training completed")
    except Exception as e:
        print(f"❌ Trial {trial.number} failed during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise optuna.TrialPruned()

    # --- Evaluation ---
    print(f"[Trial {trial.number}] Evaluating...")
    try:
        metrics, _, _ = evaluate_model(trained_model, val_loader)
    except Exception as e:
        print(f"❌ Trial {trial.number} failed during evaluation: {str(e)}")
        raise optuna.TrialPruned()
    
    # Convert metrics to float safely
    def to_float(x):
        if isinstance(x, (np.ndarray, np.generic)):
            return float(x.item() if x.ndim == 0 else x[0])
        elif torch.is_tensor(x):
            return float(x.item())
        else:
            return float(x)
    
    f1 = to_float(metrics.get('f1', 0.0))
    roc_auc = to_float(metrics.get('roc_auc', 0.0))
    accuracy = to_float(metrics.get('accuracy', 0.0))
    precision = to_float(metrics.get('precision', 0.0))
    recall = to_float(metrics.get('recall', 0.0))
    
    # Clinical deployment thresholds (YFT-style)
    THRESHOLD = get('metric_threshold', 0.60)
    metric_vals = [f1, roc_auc, accuracy, precision, recall]
    valid = all(m >= THRESHOLD for m in metric_vals)
    
    # Composite score (weighted by clinical importance)
    weights = get('metric_weights', {
        'f1': 0.30,
        'roc_auc': 0.25,  # ↑ Weight AUC for Parkinson's
        'accuracy': 0.15,
        'precision': 0.15,
        'recall': 0.15
    })
    
    composite = (
        weights['f1'] * f1 +
        weights['roc_auc'] * roc_auc +
        weights['accuracy'] * accuracy +
        weights['precision'] * precision +
        weights['recall'] * recall
    )
    
    # Save metadata
    clean_metrics = {
        'f1': f1, 
        'roc_auc': roc_auc, 
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall
    }
    trial.set_user_attr("metrics", clean_metrics)
    trial.set_user_attr("composite", composite)
    trial.set_user_attr("valid", valid)
    
    # Print trial results
    print(f"[Trial {trial.number}] Results:")
    print(f"   F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, Acc: {accuracy:.4f}")
    print(f"   Composite: {composite:.4f}, Valid: {valid}")
    
    # ===== MODEL SAVING LOGIC =====
    # Save model weights if this is the best valid trial so far
    if valid:
        # Get all completed valid trials
        completed_valid_trials = [
            t for t in trial.study.trials 
            if t.state == optuna.trial.TrialState.COMPLETE 
            and t.user_attrs.get("valid", False)
        ]
        
        # This trial's value (remember: we minimize -composite)
        current_value = -composite
        
        # Check if this is the best so far
        is_best = True
        if completed_valid_trials:
            best_previous_value = min(t.value for t in completed_valid_trials)
            is_best = current_value < best_previous_value
        
        # Also save if this is the first valid trial
        if is_best or not completed_valid_trials:
            os.makedirs('weights', exist_ok=True)
            target_name = get("target_name", "disorder")
            weights_path = f'weights/best_model_{target_name}.pt'
            torch.save(trained_model.state_dict(), weights_path)
            print(f"💾 [Trial {trial.number}] New best! Saved weights to {weights_path}")
            trial.set_user_attr("weights_saved", True)
    
    # Return value for optimization
    if valid:
        return -composite  # Minimize negative composite (maximize composite)
    else:
        return 1.0 - min(metric_vals)  # Penalty based on worst metric


def tune_hyperparameters(
    train_loader, 
    val_loader, 
    feature_dim, 
    n_trials=12,  # Reduced from 50
    search_config=None, 
    target_name="disorder",
    study_name=None,
    storage=None,
    load_if_exists=False
):
    """
    Perform hyperparameter tuning with YFT-style composite scoring.
    Enhanced for 2026 SOTA BBTransformer with automatic FlashAttention and improved regularization.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        feature_dim: Input feature dimension
        n_trials: Number of trials to run (reduced to 12)
        search_config: Configuration for search space and training
        target_name: Name of target disorder for logging
        study_name: Optional study name for Optuna
        storage: Optional Optuna storage backend
        load_if_exists: Whether to load existing study
        
    Returns:
        best_params: Best hyperparameters found
        study: Optuna study object
    """
    if search_config is None:
        search_config = {}
    
    # Add target_name to search_config for model saving
    search_config['target_name'] = target_name
    
    # Create study name if not provided
    if study_name is None:
        study_name = f"bbtransformer_v2_{target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Configure pruner for efficient search
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=search_config.get('n_startup_trials', 4),
        n_warmup_steps=search_config.get('n_warmup_steps', 0),
        interval_steps=search_config.get('interval_steps', 1)
    )

    # Create study
    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists
    )

    # Define objective with bound parameters
    def objective(trial):
        return optuna_objective(trial, train_loader, val_loader, feature_dim, search_config)

    # Run optimization
    print(f"\n{'='*80}")
    print(f"🚀 STARTING HYPERPARAMETER TUNING: {target_name}")
    print(f"{'='*80}")
    print(f"Study: {study_name}")
    print(f"Trials: {n_trials}")
    print(f"Epochs per trial: {search_config.get('epochs', 5000)}")
    print(f"Patience: {search_config.get('patience', 100)}")
    print(f"Metric Threshold: {search_config.get('metric_threshold', 0.60)}")
    print(f"{'='*80}\n")
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        catch=(Exception,)  # Continue on errors
    )

    # Analyze results
    THRESHOLD = search_config.get('metric_threshold', 0.60)
    
    # Filter valid trials
    valid_trials = [
        t for t in study.trials 
        if t.state == optuna.trial.TrialState.COMPLETE and
        t.user_attrs.get("valid", False)
    ]
    
    # Get best trial
    if valid_trials:
        best = min(valid_trials, key=lambda t: t.value)
        status = "✅ CLINICALLY VALID"
    else:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print("❌ No trials completed successfully!\n")
            return None, study
        best = min(completed, key=lambda t: t.value if t.value < 1.0 else float('inf'))
        status = "⚠️  BEST EFFORT (threshold not met)"
    
    # FIXED: Get the actual config that was used, not just trial.params
    best_params = best.params.copy()
    best_metrics = best.user_attrs.get("metrics", {})
    composite = best.user_attrs.get("composite", 0.0)
    n_params = best.user_attrs.get("n_parameters", "N/A")
    weights_saved = best.user_attrs.get("weights_saved", False)
    
    # Save results with ACTUAL hyperparameters used
    os.makedirs('weights', exist_ok=True)
    results_path = f'weights/best_params_{target_name}.json'
    
    results = {
        'study_name': study_name,
        'best_params': best_params,
        'metrics': best_metrics,
        'composite_score': float(composite),
        'n_parameters': int(n_params) if n_params != "N/A" else n_params,
        'weights_saved': weights_saved,
        'weights_path': f'weights/best_model_{target_name}.pt' if weights_saved else None,
        'model_version': 'BBTransformer v2.0 (2026 SOTA)',
        'features': [
            'Automatic FlashAttention (PyTorch 2.0+)',
            'Proper DropPath regularization',
            'Decoupled dropout streams',
            'Improved weight initialization',
            'GQA with RoPE',
            'Multi-scale fusion'
        ],
        'search_method': 'YFT-style composite scoring for clinical deployment',
        'threshold_met': bool(valid_trials),
        'total_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'valid_trials': len(valid_trials),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"{status}")
    print(f"{'='*80}")
    print(f"📊 Trial Statistics:")
    print(f"   Total:     {results['total_trials']}")
    print(f"   Completed: {results['completed_trials']}")
    print(f"   Valid:     {results['valid_trials']} (≥{THRESHOLD:.2f} all metrics)")
    print(f"   Pruned:    {results['pruned_trials']}")
    print(f"   Failed:    {results['failed_trials']}")
    
    print(f"\n🎯 Best Trial Results:")
    print(f"   Composite Score: {composite:.4f}")
    print(f"   Parameters:      {n_params:,}" if n_params != "N/A" else f"   Parameters:      {n_params}")
    if weights_saved:
        print(f"   Weights Saved:   ✅ weights/best_model_{target_name}.pt")
    
    print(f"\n📈 Performance Metrics:")
    for metric in ['f1', 'roc_auc', 'accuracy', 'precision', 'recall']:
        val = best_metrics.get(metric, 0)
        check = '✅' if val >= THRESHOLD else '❌'
        print(f"   {metric.upper():12s}: {val:.4f} {check}")
    
    print(f"\n🏆 Best Hyperparameters:")
    
    # Architecture
    print(f"\n   Architecture:")
    for k in ['embed_dim', 'num_heads', 'num_layers', 'n_kv_heads']:
        if k in best_params:
            print(f"      {k:24s}: {best_params[k]}")
    
    # Dropout
    print(f"\n   Dropout Configuration:")
    for k in sorted([k for k in best_params.keys() if 'dropout' in k]):
        print(f"      {k:24s}: {best_params[k]:.4f}")
    
    # Regularization
    print(f"\n   Regularization:")
    if 'stochastic_depth_rate' in best_params:
        print(f"      {'stochastic_depth_rate':24s}: {best_params['stochastic_depth_rate']:.4f}")
    if 'weight_decay' in best_params:
        print(f"      {'weight_decay':24s}: {best_params['weight_decay']:.2e}")
    
    # Optimizer
    print(f"\n   Optimizer:")
    if 'lr' in best_params:
        print(f"      {'lr':24s}: {best_params['lr']:.6f}")
    
    # Other params
    other_params = [k for k in best_params.keys() 
                   if k not in ['embed_dim', 'num_heads', 'num_layers', 'n_kv_heads', 
                               'lr', 'weight_decay', 'stochastic_depth_rate'] 
                   and 'dropout' not in k]
    if other_params:
        print(f"\n   Other:")
        for k in sorted(other_params):
            v = best_params[k]
            if isinstance(v, float):
                print(f"      {k:24s}: {v:.4f}")
            else:
                print(f"      {k:24s}: {v}")
    
    print(f"\n💾 Results saved to: {results_path}")
    print(f"{'='*80}\n")
    
    return best_params, study


def load_best_params(target_name="disorder", weights_dir='weights'):
    """
    Load best hyperparameters from a previous tuning run.
    
    Args:
        target_name: Name of target disorder
        weights_dir: Directory containing saved parameters
        
    Returns:
        dict: Best hyperparameters, or None if not found
    """
    results_path = os.path.join(weights_dir, f'best_params_{target_name}.json')
    
    if not os.path.exists(results_path):
        print(f"⚠️  No saved parameters found at {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"✅ Loaded parameters from {results_path}")
    print(f"   Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"   Composite Score: {results.get('composite_score', 'N/A'):.4f}")
    
    return results['best_params']

# ======================
# TUNING WORKFLOW
# ======================


def run_tuning_workflow(
    target_name: str,
    data_path: str,
    pheno_path: str,
    tuning_config: dict,
    n_trials: int = 30,
    base_batch: int = 64,
    random_seed: int = 42,
    weights_dir: str = 'weights',
    results_dir: str = 'results'
):
    """
    Executes a complete hyperparameter tuning workflow for fMRI-based classification.

    Parameters:
        target_name (str): Name of the clinical target (e.g., 'ASD', 'Parkinsons').
        data_path (str): Path to the .npz fMRI data file.
        pheno_path (str): Path to the phenotype CSV file.
        tuning_config (dict): Configuration dictionary for hyperparameter search space.
        n_trials (int): Number of Optuna trials to run.
        base_batch (int): Initial batch size; will reduce on OOM.
        random_seed (int): Random seed for reproducibility.
        weights_dir (str): Directory to save best hyperparameters.
        results_dir (str): Directory for logging/results (created but not used here).

    Returns:
        tuple: (best_hyperparameters_dict or None, optuna_study or None)
    """
    # Ensure output directories exist
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    def load_data_with_fallback():
        """Attempts to load data with decreasing batch sizes on OOM risk."""
        for batch_size in [base_batch, 32, 16]:
            try:
                print(f"   Attempting batch_size = {batch_size}...")
                loaders = prepare_fmri_data(
                    data_path=data_path,
                    pheno_path=pheno_path,
                    target_column=target_name,
                    batch_size=batch_size,
                    random_seed=random_seed
                )
                print(f"   ✅ Success with batch_size = {batch_size}")
                return loaders, batch_size
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   ❌ OOM with batch_size = {batch_size}, trying smaller...")
                    continue
                else:
                    raise e
        raise RuntimeError("All batch sizes failed due to OOM.")

    try:
        print(f"🚀 Starting hyperparameter tuning for '{target_name}'")
        print(f"📂 Data: {data_path}")

        (train_loader, val_loader, _, metadata), used_batch = load_data_with_fallback()
        feature_dim = metadata['feature_dim']

        print(f"\n✅ Data loaded:")
        print(f"   - Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
        print(f"   - Feature dim: {feature_dim} | Batch size: {used_batch}")

        best_hp, study = tune_hyperparameters(
            train_loader=train_loader,
            val_loader=val_loader,
            feature_dim=feature_dim,
            n_trials=n_trials,
            search_config=tuning_config,
            target_name=target_name
        )

        if best_hp is not None:
            print(f"\n✅ Tuning succeeded!")
            print(f"   Best composite score: {study.best_value:.4f}")
            save_path = os.path.join(weights_dir, f'best_params_{target_name}.json')
            with open(save_path, 'w') as f:
                json.dump(best_hp, f, indent=2)
            print(f"   Saved to: {save_path}")
        else:
            print(f"\n⚠️ No trial met clinical validity threshold (≥0.60).")

        return best_hp, study

    except Exception as e:
        print(f"❌ Critical error during tuning: {str(e)}")
        traceback.print_exc()
        return None, None

    finally:
        print("\n" + "=" * 80)
