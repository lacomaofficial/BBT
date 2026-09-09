# bbtransformer\trainer\pred.py


# Import Essentials
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.amp import autocast 
from .eval import evaluate_model
from ..utils import load_roi_names  

class Diagnostic:
    """
    Single subject prediction with interpretation
    """
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata
        self.device = next(model.parameters()).device
        self.model.eval()
        
        # Validate that age normalization stats are available
        if 'age_mean' not in metadata or 'age_std' not in metadata:
            raise ValueError(
                "Metadata must contain 'age_mean' and 'age_std' from prepare_fmri_data. "
                "Please ensure you're using the updated prepare_fmri_data function."
            )

    def predict_single(self, fmri, age, ext):
        """
        Make prediction for a single subject
        Args:
            fmri: (490, 414) numpy array — assumed already z-scored per subject
            age: float — raw age value (e.g., 56.3)
            ext: int (0 or 1)
        Returns:
            Dictionary with prediction and confidence
        """
        # Normalize age using TRAIN SET statistics
        age_norm = (age - self.metadata['age_mean']) / self.metadata['age_std']
        
        # Convert to tensors
        fmri_tensor = torch.from_numpy(fmri).unsqueeze(0).float().to(self.device)
        age_tensor = torch.tensor([age_norm], dtype=torch.float32).to(self.device)
        ext_tensor = torch.tensor([ext], dtype=torch.long).to(self.device)

        with torch.no_grad(), autocast(device_type=self.device.type):
            logits = self.model(fmri_tensor, age_tensor, ext_tensor)
            prob = torch.sigmoid(logits).item()

        prediction = 1 if prob > 0.5 else 0
        confidence = abs(prob - 0.5) * 2  # Scale to [0, 1]

        return {
            'prediction': prediction,
            'probability': prob,
            'confidence': confidence,
            'interpretation': self._interpret_prediction(prob, confidence)
        }

    def _interpret_prediction(self, prob, confidence):
        """Generate human-readable interpretation"""
        if confidence < 0.3:
            strength = "LOW CONFIDENCE"
            recommendation = "Prediction uncertain - recommend additional assessment"
        elif confidence < 0.6:
            strength = "MODERATE CONFIDENCE"
            recommendation = "Fair confidence in prediction"
        else:
            strength = "HIGH CONFIDENCE"
            recommendation = "Strong confidence in prediction"
        direction = "POSITIVE" if prob > 0.5 else "NEGATIVE"
        return {
            'direction': direction,
            'strength': strength,
            'recommendation': recommendation
        }

    def predict_batch(self, data_loader):
        """
        Make predictions for a batch of subjects
        Returns:
            Dictionary with predictions, probabilities, and metrics
        """
        all_preds = []
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for fmri, age, ext, labels in tqdm(data_loader, desc="Predicting"):
                fmri = fmri.to(self.device)
                age = age.to(self.device)
                ext = ext.to(self.device)
                with autocast(device_type=self.device.type):
                    logits = self.model(fmri, age, ext)
                    probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
        return {
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probs),
            'targets': np.array(all_targets)
        }


