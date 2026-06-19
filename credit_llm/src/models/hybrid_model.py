"""
Hybrid credit risk model combining structured features with LLM risk scores.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import logging
import joblib

logger = logging.getLogger(__name__)

class HybridCreditModel:
    """Hybrid model combining structured features with LLM risk scores."""
    
    def __init__(
        self,
        model_params: Optional[Dict] = None,
        random_state: int = 42,
        use_smote: bool = True
    ):
        self.model_params = model_params or {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.05,
            'class_weight': 'balanced',
            'random_state': random_state,
            'verbose': -1
        }
        self.model = LGBMClassifier(**self.model_params)
        self.scaler = StandardScaler()
        self.use_smote = use_smote
        self.features = []
        self.random_state = random_state
    
    def prepare_features(self, df: pd.DataFrame, include_llm: bool = True) -> pd.DataFrame:
        """Prepare features for model training."""
        df = df.copy()
        
        # Encode categorical
        categorical_cols = ['gender', 'region', 'employment_status', 'loan_purpose']
        for col in categorical_cols:
            if col in df.columns:
                df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        
        # Feature groups
        feature_groups = {
            'Demographic': ['age', 'gender_encoded', 'region_encoded', 'employment_encoded'],
            'Traditional Credit': ['credit_score', 'existing_loans'],
            'Alternative Data': [
                'has_mobile_money', 'mobile_money_tenure_months', 'mobile_money_avg_balance_ghs',
                'has_utility_account', 'utility_on_time_rate', 'utility_tenure_months',
                'has_bank_account'
            ],
            'Loan Details': ['loan_amount_ghs', 'loan_term_months', 'loan_purpose_encoded']
        }
        
        if include_llm and 'llm_risk_score' in df.columns:
            feature_groups['LLM Features'] = ['llm_risk_score']
        
        # Combine features
        features = []
        for group, feats in feature_groups.items():
            available = [f for f in feats if f in df.columns]
            features.extend(available)
        
        X = df[features].copy()
        
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        self.features = features
        return X
    
    def train(self, df: pd.DataFrame, target_col: str = 'default', 
              include_llm: bool = True, test_size: float = 0.2) -> Dict:
        """Train the hybrid model."""
        
        logger.info("Preparing features...")
        X = self.prepare_features(df, include_llm)
        y = df[target_col]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train):,}")
        logger.info(f"Test set: {len(X_test):,}")
        logger.info(f"Default rate (train): {y_train.mean():.2%}")
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # SMOTE
        if self.use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            logger.info(f"After SMOTE: Defaults={y_train_resampled.sum():,}, Non-defaults={len(y_train_resampled)-y_train_resampled.sum():,}")
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
        
        # Train
        logger.info("Training LightGBM model...")
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob)
        }
        
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"AUC-ROC: {results['auc_roc']:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        self.results = results
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, path: str = "outputs/models/hybrid_model.pkl"):
        """Save model and scaler."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'model_params': self.model_params,
            'results': self.results
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str = "outputs/models/hybrid_model.pkl"):
        """Load model and scaler."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.features = data['features']
        self.model_params = data['model_params']
        self.results = data.get('results', {})
        logger.info(f"Model loaded from {path}")