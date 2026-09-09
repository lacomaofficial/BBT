# Biomarker Discovery 

<br>

> **A sequential multi-task framework for identifying transdiagnostic neuroimaging biomarkers using interpretable deep learning**

This repository implements a principled pipeline for **data-driven biomarker discovery** in large-scale neuroimaging cohorts. Leveraging the **BBTransformer** architecture and **permutation-based feature importance**, we rank the contribution of 414 cortical and subcortical Regions of Interest (ROIs) to diagnostic classification across 12 neurological and psychiatric conditions. Critically, the system employs **sequential transfer learning**: validated models from earlier disorders serve as pretrained weights for subsequent tasks, promoting stability and generalizability of identified biomarkers.



## 1. Overview

Neuroimaging biomarkers hold promise for objective diagnosis, prognosis, and mechanistic insight—but their discovery is hindered by high dimensionality, noise, and disorder heterogeneity. Our approach addresses these challenges through:

- **Model-agnostic interpretability**: Permutation importance quantifies ROI relevance without assumptions about model internals.
- **Cross-disorder knowledge transfer**: Sequential fine-tuning stabilizes feature rankings across related pathologies.
- **Rigorous validation gating**: Biomarker extraction only proceeds if model performance exceeds a composite threshold (default: ≥0.65).


<br>

## 2. Methodology: Permutation Importance

Permutation importance measures the drop in predictive performance when a single feature’s association with the target is destroyed via random shuffling. Formally, for ROI \(i\):


- \(N = 50\) permutations per ROI (configurable),
- Shuffling is performed **across subjects** within mini-batches to preserve temporal structure while breaking label-ROI coupling.

A larger importance score indicates greater reliance of the model on that ROI for accurate prediction.

> **Why permutation?** Unlike gradient- or attention-based methods, permutation importance is **model-agnostic**, **nonlinearly aware**, and directly tied to predictive utility—not internal activation magnitude.

<br>

## 3. Supported Disorders (ICD/Phenotypic Groups)


| Category | Conditions |
|--------|-----------|
| **Neurodevelopmental** | Autism Spectrum Disorder (ASD) |
| **Dementia Subtypes** | Developmental Dementia, Psychopathology Dementia |
| **Organic Disorders** | Organic Mental Disorder, Cerebrovascular, Inflammatory/Infectious |
| **Psychosis & Mood** | Schizophrenia Spectrum, Bipolar (F31) |
| **Movement & Seizure** | Epilepsy/Status Epilepticus, Parkinson’s & Other Movement, Multiple Sclerosis |

<br>

## 4. Implementation Details

### 4.1 Core Functions

#### `calculate_permutation_importance` (`bbtransformer/trainer/rank.py`)
- **Inputs**: Trained model, validation DataLoader, `feature_dim=414`, `n_repeats=50`, `metric='f1'`
- **Key Features**:
  - GPU-accelerated shuffling via `torch.randperm`
  - Mixed-precision inference (`torch.cuda.amp.autocast`)
  - Batch-wise permutation to maintain memory efficiency
- **Output**: `(414,)` NumPy array of mean importance scores

#### `save_top_importance_to_csv`
- Merges importance scores with ROI metadata (name, network, subsystem)
- Exports top-*N* (default: 30) to `results/importance_<disorder>.csv`

### 4.2 Configuration Parameters

| Parameter | Default | Description |
|---------|--------|-------------|
| `importance_n_repeats` | 50 | Permutations per ROI; ↑ stability, ↑ runtime |
| `importance_metric` | `'f1'` | Scoring metric; `'f1'` preferred for imbalanced labels |
| `min_composite` | 0.65 | Performance gate for biomarker extraction |

<br>

## 5. Output Artifacts

For each validated disorder, the following are saved in `results/`:

**`importance_<disorder>.csv`**  
   → Top-*N* ROIs: `roi_index`, `roi_name`, `functional_system`, `subsystem`, `importance`



<br>

## 6. Interpretation Guidelines

- **High importance**: ROI contains discriminative signal; its disruption degrades performance.
- **Low/negative importance**: ROI is non-informative or introduces noise.


> **Caution**: Importance ≠ causality. These are *predictive* biomarkers—not necessarily etiological drivers.



<br>

<br>

## Tables



### **1. Autism Spectrum Disorder (ASD)**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| R_p9-46v_ROI     | Right      | Posterior Ventral Prefrontal Area 9-46 | Frontoparietal    | VLPFC            | 0.0428     |
| PUT-VA-rh        | Right      | Putamen (Ventral Anterior)           | BasalGanglia        | Striatum         | 0.0389     |
| R_p10p_ROI       | Right      | Posterior Polar Area 10              | DefaultMode         | Prefrontal       | 0.0379     |
| L_FOP3_ROI       | Left       | Frontal Opercular Area 3             | CinguloOpercular    | FrontalOperculum | 0.0369     |
| R_PEF_ROI        | Right      | Parietal Eye Field                   | DorsalAttention     | Parietal         | 0.0362     |
| R_52_ROI         | Right      | Parainsular Area 52                  | Language            | Parainsular      | 0.0348     |
| R_PGs_ROI        | Right      | Superior Angular Gyrus (Area PG)     | DefaultMode         | Parietal         | 0.0344     |
| L_PFcm_ROI       | Left       | Centromedial Inferior Parietal Area PF | DefaultMode       | Parietal         | 0.0317     |
| L_PI_ROI         | Left       | Parieto-Insular Vestibular Cortex    | VentralAttention    | Vestibular       | 0.0316     |
| L_45_ROI         | Left       | Inferior Frontal Pars Triangularis (Broca's Area) | Language      | Broca            | 0.0314     |

<br>

### **2. NervousSystem_Dementia_Developmental**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| R_PF_ROI         | Right      | Supramarginal Gyrus (Area PF)        | DefaultMode         | Parietal         | 0.0662     |
| R_11l_ROI        | Right      | Lateral Orbitofrontal Area 11        | Limbic              | Orbitofrontal    | 0.0662     |
| L_5mv_ROI        | Left       | Medioventral Area 5                  | Somatosensory       | Association      | 0.0662     |
| L_47m_ROI        | Left       | Medial Orbitofrontal Area 47         | Limbic              | Orbitofrontal    | 0.0662     |
| R_PCV_ROI        | Right      | Precuneus Visual Area                | DefaultMode         | Precuneus        | 0.0650     |
| L_8C_ROI         | Left       | Prefrontal Area 8C                   | DefaultMode         | Prefrontal       | 0.0649     |
| L_PHT_ROI        | Left       | Posterior Parahippocampal Area       | Limbic              | Parahippocampal  | 0.0649     |
| R_7PL_ROI        | Right      | Posterolateral Area 7P               | DorsalAttention     | Parietal         | 0.0635     |
| R_7Pm_ROI        | Right      | Medial Area 7P                       | DorsalAttention     | Parietal         | 0.0635     |
| L_MIP_ROI        | Left       | Medial Intraparietal Area            | DorsalAttention     | Parietal         | 0.0635     |

<br>

### **3. Psychopathology_Dementia**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| R_a24_ROI        | Right      | Anterior Cingulate Area 24           | CinguloOpercular    | AnteriorCingulate| 0.0747     |
| R_6a_ROI         | Right      | Anterior Premotor Area 6a            | Motor               | Premotor         | 0.0747     |
| R_7PL_ROI        | Right      | Posterolateral Area 7P               | DorsalAttention     | Parietal         | 0.0747     |
| L_p32_ROI        | Left       | Posterior Anterior Cingulate Area 32 | CinguloOpercular    | AnteriorCingulate| 0.0747     |
| R_p24_ROI        | Right      | Posterior Cingulate Area 24          | CinguloOpercular    | MidCingulate     | 0.0747     |
| L_24dv_ROI       | Left       | Ventral Anterior Cingulate Area 24d  | CinguloOpercular    | AnteriorCingulate| 0.0747     |
| R_PEF_ROI        | Right      | Parietal Eye Field                   | DorsalAttention     | Parietal         | 0.0731     |
| L_a32pr_ROI      | Left       | Anterior Pregenual Cingulate Area 32 | CinguloOpercular    | AnteriorCingulate| 0.0731     |
| L_V6_ROI         | Left       | Visual Area 6 (V6)                   | Visual              | Dorsal           | 0.0731     |
| R_IP0_ROI        | Right      | Intraparietal Area 0                 | DorsalAttention     | Parietal         | 0.0731     |

<br>

### **4. Psychopathology_Organic_Mental_Disorder**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| L_45_ROI         | Left       | Inferior Frontal Pars Triangularis (Broca's Area) | Language      | Broca            | 0.0283     |
| L_v23ab_ROI      | Left       | Ventral Posterior Cingulate Area 23ab| DefaultMode         | PosteriorCingulate| 0.0241    |
| R_p10p_ROI       | Right      | Posterior Polar Area 10              | DefaultMode         | Prefrontal       | 0.0231     |
| L_10v_ROI        | Left       | Ventral Frontopolar Area 10          | DefaultMode         | Prefrontal       | 0.0228     |
| R_TE1a_ROI       | Right      | Anterior Inferotemporal Area TE1     | Temporal            | Inferotemporal   | 0.0186     |
| L_V8_ROI         | Left       | Visual Area 8 (V8)                   | Visual              | Ventral          | 0.0185     |
| R_V2_ROI         | Right      | Secondary Visual Cortex (V2)         | Visual              | EarlyRetinotopic | 0.0178     |
| R_A4_ROI         | Right      | Auditory Association Area 4          | Auditory            | Association      | 0.0159     |
| L_7PL_ROI        | Left       | Posterolateral Area 7P               | DorsalAttention     | Parietal         | 0.0133     |
| R_11l_ROI        | Right      | Lateral Orbitofrontal Area 11        | Limbic              | Orbitofrontal    | 0.0126     |

<br>

### **5. NervousSystem_Inflammatory_Infectious**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| R_11l_ROI        | Right      | Lateral Orbitofrontal Area 11        | Limbic              | Orbitofrontal    | 0.0233     |
| R_PCV_ROI        | Right      | Precuneus Visual Area                | DefaultMode         | Precuneus        | 0.0103     |
| R_PEF_ROI        | Right      | Parietal Eye Field                   | DorsalAttention     | Parietal         | 0.0102     |
| R_8BL_ROI        | Right      | Lateral Prefrontal Area 8B           | DefaultMode         | Prefrontal       | 0.0087     |
| L_V1_ROI         | Left       | Primary Visual Cortex (V1)           | Visual              | EarlyRetinotopic | 0.0059     |
| L_47m_ROI        | Left       | Medial Orbitofrontal Area 47         | Limbic              | Orbitofrontal    | 0.0057     |
| R_FOP4_ROI       | Right      | Frontal Opercular Area 4             | CinguloOpercular    | FrontalOperculum | 0.0056     |
| R_1_ROI          | Right      | Primary Somatosensory Cortex (Area 1)| Somatosensory       | Primary          | 0.0049     |
| L_LO2_ROI        | Left       | Lateral Occipital Area 2 (LO2)       | Visual              | Lateral          | 0.0048     |
| L_IP2_ROI        | Left       | Intraparietal Area 2                 | DorsalAttention     | Parietal         | 0.0045     |

<br>

### **6. NervousSystem_Cerebrovascular**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| R_PF_ROI         | Right      | Supramarginal Gyrus (Area PF)        | DefaultMode         | Parietal         | 0.0215     |
| L_TE1p_ROI       | Left       | Posterior Inferotemporal Area TE1    | Temporal            | Inferotemporal   | 0.0203     |
| R_IFJa_ROI       | Right      | Anterior Inferior Frontal Junction   | Frontoparietal      | IFJ              | 0.0191     |
| R_55b_ROI        | Right      | Premotor Area 55b                    | Motor               | Premotor         | 0.0187     |
| L_5L_ROI         | Left       | Lateral Area 5 (Superior Parietal Lobule) | Somatosensory    | Association      | 0.0183     |
| R_V8_ROI         | Right      | Visual Area 8 (V8)                   | Visual              | Ventral          | 0.0182     |
| R_LO3_ROI        | Right      | Lateral Occipital Area 3 (LO3)       | Visual              | Lateral          | 0.0181     |
| R_ProS_ROI       | Right      | Prosubiculum                         | Limbic              | MedialTemporal   | 0.0181     |
| R_TPOJ1_ROI      | Right      | Temporoparietal Occipital Junction 1 | VentralAttention    | Junction         | 0.0174     |
| L_PH_ROI         | Left       | Parahippocampal Area                 | Limbic              | Parahippocampal  | 0.0173     |

<br>

### **7. ICD_F31_Bipolar**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| L_IP2_ROI        | Left       | Intraparietal Area 2                 | DorsalAttention     | Parietal         | 0.0670     |
| R_p10p_ROI       | Right      | Posterior Polar Area 10              | DefaultMode         | Prefrontal       | 0.0610     |
| R_p9-46v_ROI     | Right      | Posterior Ventral Prefrontal Area 9-46 | Frontoparietal    | VLPFC            | 0.0381     |
| R_PGs_ROI        | Right      | Superior Angular Gyrus (Area PG)     | DefaultMode         | Parietal         | 0.0213     |
| R_PFm_ROI        | Right      | Medial Parietal Area PF              | DefaultMode         | Parietal         | 0.0213     |
| L_LO2_ROI        | Left       | Lateral Occipital Area 2 (LO2)       | Visual              | Lateral          | 0.0213     |
| R_46_ROI         | Right      | Dorsolateral Prefrontal Area 46      | Frontoparietal      | DLPFC            | 0.0198     |
| R_PGp_ROI        | Right      | Posterior Parietal Area PG           | DefaultMode         | Parietal         | 0.0137     |
| R_IP0_ROI        | Right      | Intraparietal Area 0                 | DorsalAttention     | Parietal         | 0.0122     |
| R_11l_ROI        | Right      | Lateral Orbitofrontal Area 11        | Limbic              | Orbitofrontal    | 0.0122     |

<br>

### **8. Psychopathology_Schizophrenia_Spectrum**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| R_IPS1_ROI       | Right      | Intraparietal Sulcus Area 1          | DorsalAttention     | Parietal         | 0.1111     |
| L_FOP1_ROI       | Left       | Frontal Opercular Area 1             | CinguloOpercular    | FrontalOperculum | 0.1089     |
| R_V2_ROI         | Right      | Secondary Visual Cortex (V2)         | Visual              | EarlyRetinotopic | 0.1067     |
| L_VMV2_ROI       | Left       | Ventromedial Visual Area 2 (VMV2)    | Visual              | Ventral          | 0.1067     |
| L_V3A_ROI        | Left       | Visual Area 3A (V3A)                 | Visual              | Dorsal           | 0.1067     |
| L_a32pr_ROI      | Left       | Anterior Pregenual Cingulate Area 32 | CinguloOpercular    | AnteriorCingulate| 0.1067     |
| R_IFSa_ROI       | Right      | Anterior Inferior Frontal Sulcus     | Frontoparietal      | IFS              | 0.1044     |
| L_TE1m_ROI       | Left       | Middle Inferotemporal Area TE1       | Temporal            | Inferotemporal   | 0.1044     |
| R_7PC_ROI        | Right      | Superior Parietal Area 7PC           | DorsalAttention     | Parietal         | 0.1044     |
| R_IFJa_ROI       | Right      | Anterior Inferior Frontal Junction   | Frontoparietal      | IFJ              | 0.1044     |

<br>

### **9. NervousSystem_Epilepsy_Status_Epilepticus**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| L_MIP_ROI        | Left       | Medial Intraparietal Area            | DorsalAttention     | Parietal         | 0.0311     |
| R_11l_ROI        | Right      | Lateral Orbitofrontal Area 11        | Limbic              | Orbitofrontal    | 0.0302     |
| L_TPOJ3_ROI      | Left       | Temporoparietal Occipital Junction 3 | VentralAttention    | Junction         | 0.0199     |
| R_TE1m_ROI       | Right      | Middle Inferotemporal Area TE1       | Temporal            | Inferotemporal   | 0.0197     |
| L_10d_ROI        | Left       | Dorsal Frontopolar Area 10           | DefaultMode         | Prefrontal       | 0.0189     |
| R_8BL_ROI        | Right      | Lateral Prefrontal Area 8B           | DefaultMode         | Prefrontal       | 0.0185     |
| R_PGp_ROI        | Right      | Posterior Parietal Area PG           | DefaultMode         | Parietal         | 0.0184     |
| R_10d_ROI        | Right      | Dorsal Frontopolar Area 10           | DefaultMode         | Prefrontal       | 0.0179     |
| R_EC_ROI         | Right      | Entorhinal Cortex                    | Limbic              | MedialTemporal   | 0.0169     |
| L_s6-8_ROI       | Left       | Superior Transitional Area 6-8       | Frontoparietal      | Premotor         | 0.0158     |

<br>

### **10. NervousSystem_Parkinsons_Other_Movement**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| L_8C_ROI         | Left       | Prefrontal Area 8C                   | DefaultMode         | Prefrontal       | 0.0447     |
| R_11l_ROI        | Right      | Lateral Orbitofrontal Area 11        | Limbic              | Orbitofrontal    | 0.0388     |
| R_TPOJ1_ROI      | Right      | Temporoparietal Occipital Junction 1 | VentralAttention    | Junction         | 0.0346     |
| L_IPS1_ROI       | Left       | Intraparietal Sulcus Area 1          | DorsalAttention     | Parietal         | 0.0343     |
| L_45_ROI         | Left       | Inferior Frontal Pars Triangularis (Broca's Area) | Language      | Broca            | 0.0328     |
| THA-DP-lh        | Left       | Thalamus (Dorsal Posterior / Pulvinar) | Thalamus          | Thalamus         | 0.0312     |
| R_AVI_ROI        | Right      | Anterior Ventral Insular Area        | CinguloOpercular    | Insula           | 0.0312     |
| L_p32_ROI        | Left       | Posterior Anterior Cingulate Area 32 | CinguloOpercular    | AnteriorCingulate| 0.0309     |
| L_s6-8_ROI       | Left       | Superior Transitional Area 6-8       | Frontoparietal      | Premotor         | 0.0303     |
| L_v23ab_ROI      | Left       | Ventral Posterior Cingulate Area 23ab| DefaultMode         | PosteriorCingulate| 0.0287    |

<br>

### **11. NervousSystem_Multiple_Sclerosis_Other_Demyelinating**

| ROI Name        | Hemisphere | Region Full Name                     | Functional System   | Sub-system      | Importance |
|------------------|------------|--------------------------------------|---------------------|------------------|------------|
| R_p10p_ROI       | Right      | Posterior Polar Area 10              | DefaultMode         | Prefrontal       | 0.0235     |
| R_8BL_ROI        | Right      | Lateral Prefrontal Area 8B           | DefaultMode         | Prefrontal       | 0.0160     |
| R_11l_ROI        | Right      | Lateral Orbitofrontal Area 11        | Limbic              | Orbitofrontal    | 0.0124     |
| R_PCV_ROI        | Right      | Precuneus Visual Area                | DefaultMode         | Precuneus        | 0.0094     |
| L_PF_ROI         | Left       | Supramarginal Gyrus (Area PF)        | DefaultMode         | Parietal         | 0.0083     |
| R_s6-8_ROI       | Right      | Superior Transitional Area 6-8       | Frontoparietal      | Premotor         | 0.0035     |
| R_TE2p_ROI       | Right      | Posterior Inferotemporal Area TE2    | Temporal            | Inferotemporal   | 0.0024     |
| L_9a_ROI         | Left       | Anterior Prefrontal Area 9           | DefaultMode         | Prefrontal       | 0.0022     |
| R_MST_ROI        | Right      | Medial Superior Temporal Area (MST)  | Visual              | Dorsal           | 0.0022     |
| L_STSdp_ROI      | Left       | Dorsal Posterior Superior Temporal Sulcus | VentralAttention | Temporal         | 0.0022     |

---




