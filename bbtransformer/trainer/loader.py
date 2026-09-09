# bbtransformer\trainer\loader.py

# Import Essentials
import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Any, List
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d



# ==============================================
# DATA PREPARATION
# ==============================================

def prepare_fmri_data(
    data_path='ukbb_features.npz',
    pheno_path='ukbb_pheno.csv',
    target_column='Target',
    age_column='Age',
    ext_column='Sex',
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    batch_size=32,
    random_seed=42,
    num_workers=4,
    pin_memory=True,
):
    """Prepare fMRI data loaders with stratified splits and age normalization."""
    
    # Load and align data
    npz = np.load(data_path)
    fMRI = npz['data'].astype(np.float32)
    ids = npz['subject_ids'].astype(str)
    
    pheno = pd.read_csv(pheno_path, dtype={'eid': str})
    if 'eid' in pheno.columns:
        pheno = pheno.set_index('eid')
    
    # Validate and align
    required_cols = [target_column, age_column, ext_column]
    missing = [c for c in required_cols if c not in pheno.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    if not set(ids).issubset(pheno.index):
        raise ValueError("Subject ID mismatch between fMRI and phenotype")
    
    pheno = pheno.loc[ids]
    
    # Extract and clean data
    y = pheno[target_column].values.astype(np.float32)
    age = pheno[age_column].values.astype(np.float32)
    ext = pheno[ext_column].values.astype(np.int64)
    
    valid = np.isfinite(y) & ((y == 0) | (y == 1)) & np.isfinite(age) & ((ext == 0) | (ext == 1))
    fMRI, y, age, ext = fMRI[valid], y[valid], age[valid], ext[valid]
    
    print(f"Cohort: {len(y)} subjects ({int(y.sum())} cases, {len(y)-int(y.sum())} controls, {y.mean():.1%} prevalence)")
    
    # Stratified splits
    idx = np.arange(len(fMRI))
    train_idx, temp_idx = train_test_split(idx, test_size=val_split+test_split, stratify=y, random_state=random_seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_split/(val_split+test_split), 
                                         stratify=y[temp_idx], random_state=random_seed) if test_split > 0 else (temp_idx, np.array([]))
    
    # Normalize age using train stats
    age_mean, age_std = age[train_idx].mean(), age[train_idx].std() + 1e-8
    age = (age - age_mean) / age_std
    
    # Convert to tensors
    tensors = (torch.from_numpy(fMRI), torch.from_numpy(age).float(), 
               torch.from_numpy(ext).long(), torch.from_numpy(y).float())
    
    # Create loaders
    def make_loader(indices, shuffle=False):
        if len(indices) == 0:
            return None
        ds = TensorDataset(*[t[indices] for t in tensors])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    
    loaders = (make_loader(train_idx, True), make_loader(val_idx), make_loader(test_idx))
    
    metadata = {
        'target': target_column,
        'n_total': len(y),
        'n_positive': int(y.sum()),
        'prevalence': float(y.mean()),
        'feature_dim': fMRI.shape[-1],
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'age_mean': float(age_mean),
        'age_std': float(age_std),
    }
    
    print(f"Splits → Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return *loaders, metadata



