"""
Baseline models for credit risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import logging
import joblib

logger = logging.getLogger(__name__)

class BaselineModels:
    """Train and evaluate baseline models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=random_state, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1,
                verbose=-1
            )
        }
        self.results = {}
        self.trained_models = {}
        self.scaler = StandardScaler()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling."""
        df = df.copy()
        
        # Encode categorical
        categorical_cols = ['gender', 'region', 'employment_status', 'loan_purpose']
        for col in categorical_cols:
            if col in df.columns:
                df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        
        # Select features
        features = [
            'age', 'gender_encoded', 'region_encoded', 'employment_encoded',
            'credit_score', 'existing_loans',
            'has_mobile_money', 'mobile_money_tenure_months', 'mobile_money_avg_balance_ghs',
            'has_utility_account', 'utility_on_time_rate', 'utility_tenure_months',
            'has_bank_account', 'loan_amount_ghs', 'loan_term_months', 'loan_purpose_encoded'
        ]
        
        # Keep only available features
        features = [f for f in features if f in df.columns]
        X = df[features].copy()
        
        # Handle missing
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def train_all(self, df: pd.DataFrame, target_col: str = 'default') -> Dict:
        """Train all baseline models."""
        
        logger.info("Preparing features...")
        X = self.prepare_features(df)
        y = df[target_col]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        logger.info(f"Training set: {len(X_train):,}, Test set: {len(X_test):,}")
        logger.info(f"After SMOTE: {len(X_train_resampled):,}")
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_resampled, y_train_resampled)
            self.trained_models[name] = model
            
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_prob)
            }
            
            logger.info(f"  Accuracy: {self.results[name]['accuracy']:.4f}")
            logger.info(f"  AUC-ROC: {self.results[name]['auc_roc']:.4f}")
        
        return self.results
    
    def get_best_model(self) -> tuple:
        """Get the best performing model."""
        best_name = max(self.results, key=lambda x: self.results[x]['auc_roc'])
        return best_name, self.trained_models[best_name]
    
    def save_models(self, path: str = "outputs/models/"):
        """Save all trained models."""
        import os
        os.makedirs(path, exist_ok=True)
        for name, model in self.trained_models.items():
            joblib.dump(model, f"{path}/baseline_{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(self.scaler, f"{path}/baseline_scaler.pkl")
        logger.info(f"Models saved to {path}")