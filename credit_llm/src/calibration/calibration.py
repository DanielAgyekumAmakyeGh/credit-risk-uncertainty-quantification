"""
Model calibration methods for uncertainty quantification.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class ModelCalibrator:
    """Calibrate model probabilities for better uncertainty quantification."""
    
    def __init__(self, method: str = 'sigmoid', cv_folds: int = 5):
        self.method = method
        self.cv_folds = cv_folds
        self.calibrated_model = None
        self.results = {}
    
    def calibrate(self, model, X_train, y_train):
        """Apply Platt scaling (sigmoid calibration)."""
        
        logger.info(f"Calibrating model using {self.method} with {self.cv_folds} folds...")
        
        self.calibrated_model = CalibratedClassifierCV(
            model,
            method=self.method,
            cv=self.cv_folds
        )
        self.calibrated_model.fit(X_train, y_train)
        
        logger.info("Calibration complete")
        return self.calibrated_model
    
    def evaluate_calibration(self, model, X_test, y_test):
        """Evaluate calibration quality."""
        
        # Get probabilities
        uncalibrated_probs = model.predict_proba(X_test)[:, 1]
        
        if self.calibrated_model is not None:
            calibrated_probs = self.calibrated_model.predict_proba(X_test)[:, 1]
        else:
            calibrated_probs = uncalibrated_probs
        
        # Brier scores
        brier_uncal = brier_score_loss(y_test, uncalibrated_probs)
        brier_cal = brier_score_loss(y_test, calibrated_probs)
        
        # Confidence scores
        conf_uncal = 1 - 2 * np.abs(uncalibrated_probs - 0.5)
        conf_cal = 1 - 2 * np.abs(calibrated_probs - 0.5)
        
        self.results = {
            'brier_uncalibrated': brier_uncal,
            'brier_calibrated': brier_cal,
            'improvement': (brier_uncal - brier_cal) / brier_uncal * 100,
            'confidence_uncalibrated': conf_uncal.mean(),
            'confidence_calibrated': conf_cal.mean()
        }
        
        logger.info(f"Brier Score - Uncalibrated: {brier_uncal:.4f}, Calibrated: {brier_cal:.4f}")
        logger.info(f"Improvement: {self.results['improvement']:.2f}%")
        
        return self.results
    
    def plot_calibration_curve(self, model, X_test, y_test, save_path: str = None):
        """Plot calibration curve."""
        
        # Get probabilities
        if self.calibrated_model is not None:
            probs_uncal = model.predict_proba(X_test)[:, 1]
            probs_cal = self.calibrated_model.predict_proba(X_test)[:, 1]
        else:
            probs_uncal = model.predict_proba(X_test)[:, 1]
            probs_cal = probs_uncal
        
        # Calibration curves
        prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, probs_uncal, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_test, probs_cal, n_bins=10)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(prob_pred_uncal, prob_true_uncal, 'o-', label='Uncalibrated', linewidth=2)
        plt.plot(prob_pred_cal, prob_true_cal, 's-', label='Calibrated', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {save_path}")
        
        plt.show()