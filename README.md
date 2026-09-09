# **BBT: A Multivariate Time Series Transformer for Modeling Stochastic Dynamical Systems in the Human Brain**

<br>

**Author:** *J.Zorraquin* 

**University of Augsburg:** Chair of Informatics for Medical Technologies (CIMT)



<br>

## **Abstract**

Standard resting-state functional MRI (rs-fMRI) analysis collapses the BOLD signal into static functional connectivity matrices, discarding the transient neural dynamics that may underlie individual pathology. We introduce BBT, a novel multivariate time-series architecture that operates directly on the continuous spatiotemporal trajectory of raw, whole-brain BOLD activity, treating the brain as a stochastic dynamical system rather than a static network. In this specific fMRI implementation, we preserve native spatiotemporal structure across 414 cortical and subcortical regions, while conditioning on age and biological sex as embedded covariates to disentangle diagnostic signals from demographic confounding. The architecture is built on three core innovations: dual-resolution temporal encoding (fine-grained and coarse patch streams), learned attention pooling that autonomously identifies diagnostically salient time windows, and confounder-aware decision fusion.

To overcome data scarcity and leverage shared neurobiological principles, we implement a biologically ordered transfer learning protocol: starting from an autism spectrum disorder (ASD) foundation model (ABIDE, N \= 585), we sequentially fine-tune our transformer model across 10 neurological and psychiatric conditions in the UK Biobank (N \= 2182), following a pathophysiological hierarchy from neurodevelopmental to white matter disorders. This strategy enables robust generalization, achieving F1 \= 0.68–0.92 and ROC-AUC \= 0.67–0.95. Critically, performance extends to two external, independently preprocessed cohorts: ADHD-200 (F1 \= 0.722, ROC-AUC \= 0.818) and the UCLA LA5c study (F1 \= 0.632, ROC-AUC \= 0.640). In stark contrast, the model fails on high-prevalence but biologically heterogeneous conditions, depressive episode (N \= 4,338), sleep disorders (N \= 1,104), and substance use disorders (N \= 2,276), despite their large sample sizes. This dissociation demonstrates that diagnostic learnability from rs-fMRI is gated by biological coherence, not data volume. Permutation-based feature importance reveals anatomically coherent, disorder-specific biomarker circuits aligned with known disease neurobiology. These results establish a new paradigm: disorder-specific signatures reside in spatiotemporal dynamics, not static connectivity.

**Keywords:** *stochastic dynamical systems, resting-state fMRI, multivariate time-series transformer, biomarker discovery, spatiotemporal dynamics, foundation model, neuroimaging*

<br>

<br>


<img width="1678" height="933" alt="BBT_Diagram" src="https://github.com/user-attachments/assets/c5742911-634c-4a3d-9fa6-e21b49f0a3cc" />


***Figure 1.** BBT full diagram. Three parallel streams process the raw 150 × 414 BOLD input: (1) a Global Stream with a 7-layer GQA+RoPE+SwiGLU transformer encoder; (2) a Local Patch Stream producing coarse temporal tokens integrated via cross-attention; and (3) a Confounder Stream encoding age and biological sex as learned embeddings. Temporal attention pooling produces a single diagnostic embedding passed to a final MLP classifier.*

<br>




<br>

## Installation

```bash
pip install git+https://github.com/cimt-unia/BBTransformer.git
```

<br>

## Datasets standardization

| Dataset        | Conditions                              | Atlas              | Preprocessing      | Role               |
| :------------- | :-------------------------------------- | :----------------- | :----------------- | :----------------- |
| **ABIDE**      | ASD vs Controls                         | Glasser+Tian (414) | C-PAC              | Foundation Model   |
| **UK Biobank** | ICD F32, G20, G40, etc. (10 conditions) | Glasser+Tian (414) | Official UKB Pipeline | Transfer Learning  |
| **ADHD-200**   | ADHD vs Controls                        | Glasser+Tian (414) | Athena (AFNI/FSL)  | External Validation |
| **UCLA LA5c**  | Schizophrenia, Bipolar, ADHD vs Controls | Glasser+Tian (414) | fMRIPrep v0.4.4    | External Validation |

> **Note**: All datasets are standardized to **150 timepoints @ 2.0s TR** across 414 ROIs via cubic spline interpolation and per-subject z-scoring before model input

<br>




<br>

## Fine-tune (Usage Example)

```python
"""Fine-tune BBTransformer on ASD vs ADHD cohort."""

import logging
from pathlib import Path

from bbtransformer.trainer.exe import run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# BASE_DIR / "/absolute/path" produces an invalid double path.
# Use the absolute path directly since it is already complete.
DATA_PATH = "/mnt/movement/users/jaizor/xtra/data/fmri/chrt/fmri_ASD_vs_ADHD.npz"
PHENO_PATH = "/mnt/movement/users/jaizor/xtra/data/fmri/chrt/pheno_ASD_vs_ADHD.csv"
PRETRAINED_WEIGHTS = "/mnt/movement/users/jaizor/xtra/notebooks/BBT/notebooks/_/master_file/weights/weights_modex.pth"

results = run_analysis(
    # --- DATA IDENTIFICATION ---
    target_column="target_label",
    project_name="ASD_vs_ADHD_finetune",  
    data_path=DATA_PATH,
    pheno_path=PHENO_PATH,
    base_dir=None,

    # --- PROJECT STRUCTURE ---
    project_root=".",
    use_pretrained=True,
    pretrained_weight_file=PRETRAINED_WEIGHTS, 

    # --- TRAINING HYPERPARAMETERS ---
    training_config={
        "epochs": 500,
        "lr": 1e-5,
        "weight_decay": 1.14e-06,
        "patience": 50,
    },
    early_stop_metric="f1",
    use_focal_loss=False,

    # --- MODEL ARCHITECTURE ---
    model_config=None,

    # --- IMPORTANCE ANALYSIS ---
    compute_importance=True,
    importance_metric="loss",
    importance_n_repeats=2,

    # --- REPRODUCIBILITY AND HARDWARE ---
    random_seed=42,
    device=None,
    batch_size=64,  
    # --- OUTPUT CONTROL ---
    save_plots=True,
    save_json=True,
)

logging.info(
    "Pipeline complete | F1: %.4f | AUC: %.4f | Project: %s",
    results["metrics"]["f1"],
    results["metrics"]["roc_auc"],
    results["project_name"],
)
```

<br>

<br>

<img width="1785" height="1535" alt="image" src="https://github.com/user-attachments/assets/bb162441-5f1f-43ee-b01b-db0df98a64dd" />

#

<br>

<img width="1485" height="735" alt="image" src="https://github.com/user-attachments/assets/c38b0d9d-52e1-426c-bd89-724d3c22f273" />

#

<br>

<img width="1484" height="884" alt="image" src="https://github.com/user-attachments/assets/798b4f34-bd00-49c3-9be9-d76a55afc7e0" />




<br>



## Key Features

### Architecture
-   **Rotary Positional Embeddings** for temporal awareness
-   **Grouped-Query Attention (GQA)** for memory efficiency
-   **Multi-scale fusion** via patch embedding + cross-attention
-   **Stochastic Depth** (DropPath) for regularization

### Training
-   **Ranger21 optimizer** with internal LR scheduling
-   **Adaptive Focal Loss** for class imbalance
-   **Early stopping** on F1 or validation loss
-   **Mixed precision** training via `torch.amp`


<br>


### Details

The simplest way to run a full analysis pipeline (training + evaluation + biomarker discovery) is via `run_analysis`:

### Output Files
When `save_plots=True` and `save_json=True`, the following are generated in `results/`:
-   `{target}_results.png`: ROC Curve, Precision-Recall, Confusion Matrix, Calibration Curve
-   `{target}_results.json`: Metrics summary + top 10 important ROIs with anatomical names
-   `importance_{target}.csv`: Ranked list of all 414 brain regions



<br>

### 📁 Project Structure

```
BBT/
├── bbtransformer/          # Core package (installable)
│   ├── __init__.py         # Public API exports
│   ├── model.py            # BBTransformer, create_bbtransformer
│   ├── utils.py            # Weight I/O, ROI metadata loading
│   └── trainer/            # exe, train, eval, rank, pred, tune, loader
├── notebooks/              # Analysis & tutorials
├── preprocessing/          # Dataset-specific curation
└── weights/                # Pretrained & fine-tuned model weights
```






<br>




### ⚙️ Default Configuration

Best hyperparameters from clinically validated Optuna tuning [2]:

```python
DEFAULT_HP = {
    'feature_dim': 414,
    'embed_dim': 512,
    'num_heads': 16,
    'num_layers': 7,
    'n_kv_heads': 4,
    'dropout_input': 0.17,
    'dropout_classifier': 0.04,
    'patch_size': 3,
    'stochastic_depth_rate': 0.10,
    'embed_dim_age': 32,
    'embed_dim_ext': 16,
}
```

<br>




<br>

## Model Architecture

BBTransformer processes **150 × 414** fMRI time series through three integrated streams [2]:

1.  **Primary Temporal Stream**:
    -   7 transformer layers
    -   512-dim embeddings, 16 query heads, 4 KV heads (GQA)
    -   Rotary position embeddings (RoPE), RMSNorm, SwiGLU
2.  **Local Patch Stream**:
    -   Patch size 3 with cross-attention fusion
    -   Captures short-scale temporal dynamics
3.  **Temporal Attention Pooling**:
    -   Learned weighting of timepoints for diagnostic decisions
    -   Integrates confounder embeddings (Age, Sex)

Final output: Single probability via sigmoid for binary classification.


<br>



