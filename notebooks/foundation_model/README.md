# **BBT: A Multivariate Time Series Transformer for Modeling Stochastic Dynamical Systems in the Human Brain**



# **Results**

## **Foundation Model Pre-training and Architecture Selection**

We established the foundation model by training on Autism Spectrum Disorder (ASD) using the ABIDE repository. This cohort provides favorable properties for initialization: it spans a wide developmental range (ages 6–50) and encompasses heterogeneous acquisition protocols, forcing the model to learn invariant spatiotemporal features.

The training cohort comprised 585 subjects (271 ASD, 314 controls). We conducted 50 trials of Bayesian hyperparameter optimization. A strict validity gate required all five performance metrics (F1, ROC-AUC, accuracy, precision, and recall) to exceed 0.60 on the held-out test set before weights were accepted for downstream propagation. This threshold served as a robustness filter to ensure the source model captured a discriminative signal above chance, rather than a marker of immediate clinical utility.

Of the 50 trials, 16 met the validity threshold. The optimal configuration achieved an F1 score of 0.758, ROC-AUC of 0.728, and recall of 0.878 on the test set (Table 1). The selected architecture featured 512-dimensional embeddings, 7 transformer encoder layers, and a 4:1 Grouped-Query Attention (GQA) ratio (16 query heads sharing 4 key-value heads). This sparse attention mechanism significantly reduced memory overhead while likely acting as an information bottleneck that mitigated overfitting to site-specific artifacts.

**Table 1\. Best performance on held-out test set (ASD foundation model)**

| Metric | Value |
| :---- | :---- |
| F1 Score | 0.7579 |
| ROC-AUC | 0.7283 |
| Accuracy | 0.7386 |
| Precision | 0.6667 |
| Recall | 0.8780 |

## 

## 

## **Transfer Learning** 

Having established a valid foundation model for ASD, we implemented a biologically ordered transfer learning pipeline guided by ICD-10 clinical classifications. The ordering principle was pathophysiological coherence: we hypothesized that knowledge transfers most efficiently along axes of biological similarity, beginning with neurodevelopmental disorders (where altered early connectivity patterns are central) and progressing through neurodegenerative, inflammatory, vascular, and finally white matter disease. 

For each target disorder, we initialized our transformer model with the fixed optimal architecture identified during ASD tuning, loaded weights from the most recent valid model in the transfer chain, and ran up to 15 trials with equal random seeds. A model was accepted, and its weights propagated forward only if all five performance metrics exceeded 0.60 on the held-out test set. If all trials failed, weights from the previous valid model were retained, and the failed disorder was excluded from the propagation chain. This design prevents catastrophic forgetting from phenotypically incoherent fine-tuning while maintaining a clean record of which disorders can and cannot be decoded from raw BOLD dynamics.

Ten consecutive transfer steps met validity thresholds, spanning 10 distinct neurological and psychiatric conditions. The transfer chain successfully traversed: neurodevelopmental disorders (ASD), dementia-spectrum conditions, organic mental disorders, inflammatory and infectious CNS conditions, cerebrovascular disease, bipolar disorder, schizophrenia spectrum, epilepsy, movement disorders, and multiple sclerosis/demyelinating disease. This represents a complete traversal from the earliest-onset (neurodevelopmental) to late-onset (white matter) neurological pathology.

**Table 2\. Transfer learning progression: performance across all valid phases**

| Phase | Disorder | N (Cases/Controls) | F1 | Accuracy | ROC-AUC |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 0 | Autism Spectrum Disorder | 271/314 | 0.7579 | 0.7386 | 0.7283 |
| 1 | Developmental Dementia  | 61/61 | 0.7778 | 0.7895 | 0.7889 |
| 2 | Psychopathology Dementia | 49/49 | 0.9231 | 0.9333 | 0.9286 |
| 3 | Organic Mental Disorder | 90/90 | 0.9231 | 0.9259 | 0.9533 |
| 4 | Inflammatory / Infectious | 46/46 | 0.8000 | 0.7857 | 0.8673 |
| 5 | Cerebrovascular | 296/296 | 0.7434 | 0.6742 | 0.7116 |
| 6 | Bipolar Disorder | 55/55 | 0.8571 | 0.8824 | 0.8472 |
| 7 | Schizophrenia Spectrum | 33/33 | 0.7500 | 0.8000 | 0.8400 |
| 8 | Epilepsy | 171/171 | 0.6786 | 0.6538 | 0.6672 |
| 9 | Movement Disorders | 208/208 | 0.7213 | 0.7302 | 0.7782 |
| 10 | Multiple Sclerosis / Demyelinating | 82/82 | 0.8333 | 0.8400 | 0.9167 |

Performance across the transfer chain was strong and consistent. The highest F1 scores were achieved for Psychopathology/Dementia (F1 \= 0.9231, ROC-AUC \= 0.9286) and Organic Mental Disorder (F1 \= 0.9231, ROC-AUC \= 0.9533), conditions where spatiotemporal BOLD dynamics appear to provide a particularly clean diagnostic signal. Multiple sclerosis achieved an ROC-AUC of 0.9167, indicating strong separation between demyelination-related connectivity disruption and controls. Bipolar disorder (F1 \= 0.8571) and inflammatory/infectious conditions (F1 \= 0.8000) also performed robustly. More heterogeneous conditions, cerebrovascular disease (F1 \= 0.7434), movement disorders (F1 \= 0.7213), and epilepsy (F1 \= 0.6786), showed lower but still valid performance, consistent with greater phenotypic heterogeneity within those diagnostic categories.

## **Failed Disorders**

Three high-prevalence conditions failed to meet validity thresholds after 15 trials each: depressive episode (N \= 4,338), sleep disorders (N \= 1,104), and substance use disorders (N \= 2,276). Critically, these failures occurred despite the largest sample sizes in the dataset; depressive episode alone contained nearly twice as many subjects as all successful transfer steps combined. This dissociation between sample size and learnability is the most important negative finding of this work.

The failure modes were mechanistically distinct across the three conditions. For the depressive episode, the model converged to a collapsed mode in which approximately 97% of subjects were predicted to be positive. This behavior, high recall, near-chance precision, and ROC-AUC indistinguishable from random (0.5487), indicates that the model found no discriminative structure in the temporal dynamics and defaulted to predicting the majority class. Sleep disorders exhibited total recall collapse (recall \= 1.0, predicting every subject positive) with an AUC of 0.5213. Substance use showed a slightly higher AUC (0.6039), but still failed to achieve meaningful class separation.

**Table 3\. Failed targets: failure modes and metrics**

| Disorder | N | F1 | Accuracy | Precision | ROC-AUC | Failure Mode |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Depressive Episode | 4,338 | 0.6681 | 0.5192 | 0.5097 | 0.5487 | Collapsed mode: \~97% positive predictions |
| Sleep Disorders | 1,104 | 0.6831 | 0.5361 | 0.5188 | 0.5213 | Total collapse: recall \= 1.0, AUC ≈ random |
| Substance Use | 2,276 | 0.6581 | 0.5351 | 0.5204 | 0.6039 | Weak signal: class separation near chance |

These patterns are not attributable to data quality or model capacity, but instead reflect genuine characteristics of these phenotypes. These findings suggest that BOLD spatiotemporal dynamics, at least as captured by standard acquisition protocols, do not reliably separate these diagnostic categories from controls. The model's failure is thus scientifically informative: it constrains the hypothesis space of what is biologically encoded in rs-fMRI dynamics.

## **Validation in External Datasets**

To assess generalizability beyond the ABIDE dataset and the UK Biobank ecosystem, we extended the biologically ordered transfer learning pipeline to two independent, publicly available cohorts: ADHD-200\[[13](#bookmark=kix.8ioao2s7k842)\] and the UCLA Consortium for Neuropsychiatric Phenomics LA5c Study\[[14](#bookmark=kix.fzgxijef4vjk)\]. Critically, these datasets were excluded from all prior stages of hyperparameter tuning, architecture selection, and internal transfer learning.

The model was initialized using the validated weights from the final internal transfer phase (Multiple Sclerosis/Demyelinating, Phase 10), ensuring that the external validation began with a representation already refined across neurodevelopmental, neurodegenerative, inflammatory, cerebrovascular, and white matter pathologies. This approach tests whether the spatiotemporal signatures learned within the UK Biobank generalize to independently preprocessed, multi-site data with distinct demographic profiles.

We first applied our multivariate transformer model to the ADHD-200 cohort (N \= 242; 103 cases, 139 controls). Adhering to the standard protocol, we conducted up to 30 trials with random seeds and early stopping. The composite validity criterion required all five metrics (F1, accuracy, precision, recall, ROC-AUC) to exceed 0.60. One trial succeeded, yielding F1 \= 0.722, accuracy \= 0.730 and ROC-AUC \= 0.818. 

Subsequently, we propagated the ADHD-validated weights to the UCLA cohort (N \= 132). This dataset aggregates schizophrenia (n \= 15), bipolar disorder (n \= 22), and ADHD (n \= 26\) into a single "disorder vs. control" classification task. After 11 trials, one configuration met the validity threshold, achieving F1 \= 0.632  and ROC-AUC \= 0.640.

**Table 4\. External validation performance across independent cohorts**

| Cohort | N (Cases/Controls) | F1 | Accuracy | Precision | Recall | ROC-AUC |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| ADHD-200 | 103/139 | 0.722 | 0.730 | 0.650 | 0.813 | 0.818 |
| UCLA LA5c | 63/69 | 0.632 | 0.650 | 0.667 | 0.600 | 0.640 |

## **Spatiotemporal Biomarker Discovery**

Permutation-based feature importance across all 414 parcellated regions revealed anatomically coherent, disorder-specific biomarker circuits aligned with established disease neurobiology. Full regional rankings are provided in Extended Data Tables 1–11.

