### Data Architecture

**Inputs:**
1.  **fMRI File (`*.npz`):** A compressed NumPy archive containing:
    *   `data`: `np.ndarray` of shape `(N, 150, 414)`.
    *   `subject_ids`: `np.ndarray` of shape `(N,)` containing unique string identifiers.
2.  **Phenotype File (`*.csv`):** A tabular file where each row corresponds to a subject in the fMRI array, containing an `eid` column and clinical labels.

**Outputs:**
*   **Feature Tensor:** A 3D array where every dataset shares the dimensions **150 timepoints** and **414 brain regions**.
*   **Metadata Table:** A DataFrame aligned with the tensor for supervised learning.

### Transformation Logic

1.  **Loading:** The `.npz` file is opened using `numpy.load`, and the `.csv` is opened using `pandas.read_csv`.
2.  **Shape Standardization:** All datasets (ABIDE, ADHD-200, UKB, UCLA) have been resampled to a fixed temporal length of 150 points (representing 300 seconds at a 2.0s TR) and mapped to 414 specific brain regions.
3.  **Alignment:** The `subject_ids` from the fMRI file are matched to the `eid` column in the phenotype file to ensure labels correspond to the correct brain data.

### Technical Verification & Flaw Analysis

**API Validity:**
*   `numpy.load(..., allow_pickle=True)` is required to read the string array of subject IDs stored in the `.npz` files.

**Mathematical Integrity:**
*   **Uniform Dimensions:** The output shape is consistent across all cohorts: `(N, 150, 414)`. This allows for identical model architectures regardless of the source dataset.


---

### Load Data



```python
import numpy as np
import pandas as pd

# 1. Load the files
# Replace these paths with your actual file locations
fmri_data = np.load("path/to/fmri_file.npz", allow_pickle=True)
pheno_data = pd.read_csv("path/to/pheno_file.csv")

# 2. Extract the arrays
X = fmri_data['data']          # The brain activity numbers
ids = fmri_data['subject_ids'] # The list of subject names/IDs

# 3. Check the shape
print(X.shape) 
# Output will look like: (585, 150, 414)

# 4. Understand the dimensions:
# - The first number (e.g., 585) is how many subjects are in this specific file.
# - The second number (150) is the timepoints (5 minutes of data).
# - The third number (414) is the brain regions (ROIs).
```
