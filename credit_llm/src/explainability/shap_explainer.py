"""
SHAP explainability for credit risk models.
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class ShapExplainer:
    """SHAP explainability for credit risk models."""
    
    def __init__(self, model, feature_names: list = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, X_sample):
        """Create SHAP explainer."""
        
        logger.info("Creating SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_sample)
        
        if self.feature_names is None:
            if hasattr(X_sample, 'columns'):
                self.feature_names = X_sample.columns.tolist()
            else:
                self.feature_names = [f'Feature_{i}' for i in range(X_sample.shape[1])]
        
        logger.info("SHAP explainer created")
        return self
    
    def explain(self, X_sample):
        """Generate SHAP explanations."""
        
        if self.explainer is None:
            self.create_explainer(X_sample)
        
        shap_values = self.explainer.shap_values(X_sample)
        
        # Feature importance
        importance = np.abs(shap_values).mean(axis=0)
        
        self.shap_values = shap_values
        self.importance = importance
        
        return shap_values
    
    def plot_summary(self, X_sample, save_path: str = None):
        """Plot SHAP summary."""
        
        if self.shap_values is None:
            self.explain(X_sample)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X_sample,
            feature_names=self.feature_names,
            show=False
        )
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"SHAP summary saved to {save_path}")
        
        plt.show()
    
    def plot_importance(self, save_path: str = None):
        """Plot feature importance bar chart."""
        
        if self.importance is None:
            raise ValueError("Run explain() first to get importance values.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_n = min(15, len(importance_df))
        top_features = importance_df.head(top_n)
        
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'Top {top_n} Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance saved to {save_path}")
        
        plt.show()
        return importance_df
    
    def explain_individual(self, X_sample, idx: int = 0):
        """Explain a single prediction."""
        
        if self.shap_values is None:
            self.explain(X_sample)
        
        # Prediction
        if hasattr(self.model, 'predict_proba'):
            prob = self.model.predict_proba(X_sample[idx:idx+1])[0, 1]
        else:
            prob = self.model.predict(X_sample[idx:idx+1])[0]
        
        # SHAP explanation
        shap_vals = self.shap_values[idx] if len(self.shap_values.shape) > 1 else self.shap_values[idx]
        
        # Create explanation
        if hasattr(X_sample, 'iloc'):
            values = X_sample.iloc[idx].values
        else:
            values = X_sample[idx]
        
        explanation = pd.DataFrame({
            'feature': self.feature_names,
            'value': values,
            'shap_value': shap_vals
        }).sort_values('shap_value', key=abs, ascending=False)
        
        print("\n" + "=" * 60)
        print("INDIVIDUAL LOAN EXPLANATION")
        print("=" * 60)
        print(f"Predicted Probability of Default: {prob:.2%}")
        print(f"Decision: {'REJECT' if prob > 0.5 else 'APPROVE'}")
        print("\nKey Factors:")
        
        for _, row in explanation.head(5).iterrows():
            impact = "INCREASES risk" if row['shap_value'] > 0 else "DECREASES risk"
            print(f"  • {row['feature']}: {row['value']:.3f} - {impact} ({abs(row['shap_value']):.3f})")
        
        return explanation
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance DataFrame."""
        
        if self.importance is None:
            raise ValueError("Run explain() first to get importance values.")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.importance
        }).sort_values('importance', ascending=False)