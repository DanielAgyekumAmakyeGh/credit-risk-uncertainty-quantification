"""
Run all experiments for PhD research.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.ghana_data_generator import GhanaCreditDataGenerator
from src.models.baseline_models import BaselineModels
from src.models.hybrid_model import HybridCreditModel
from src.llm.bert_finetuner import BertCreditFinetuner
from src.calibration.calibration import ModelCalibrator
from src.explainability.shap_explainer import ShapExplainer

# Setup logging
os.makedirs('outputs/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_all_experiments():
    """Run complete experiment pipeline."""
    
    logger.info("=" * 70)
    logger.info("CREDIT RISK ASSESSMENT EXPERIMENTS")
    logger.info("Daniel Agyekum Amakye - IT University of Copenhagen")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # Create directories
    os.makedirs('data/synthetic', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    # 1. Generate Ghana-specific data
    logger.info("\n[1/6] Generating Ghana-specific data...")
    generator = GhanaCreditDataGenerator(n_samples=100000)
    df = generator.generate()
    df.to_csv('data/synthetic/ghana_credit_data.csv', index=False)
    logger.info("✓ Data generated and saved to data/synthetic/ghana_credit_data.csv")
    
    # 2. Train baseline models
    logger.info("\n[2/6] Training baseline models...")
    baseline = BaselineModels()
    baseline_results = baseline.train_all(df)
    baseline_name, baseline_model = baseline.get_best_model()
    baseline.save_models('outputs/models/')
    logger.info(f"✓ Baseline best model: {baseline_name} with AUC: {baseline_results[baseline_name]['auc_roc']:.4f}")
    
    # 3. Fine-tune BERT
    logger.info("\n[3/6] Fine-tuning BERT for credit risk...")
    bert = BertCreditFinetuner(epochs=2)  # Reduced for speed
    bert.initialize()
    
    # Use subset for faster training
    df_subset = df.sample(n=min(20000, len(df)), random_state=42)
    bert_results = bert.finetune(df_subset)
    bert.save_model('outputs/models/bert_credit_model/')
    logger.info(f"✓ BERT fine-tuning complete. Test AUC: {bert_results.get('test_auc', 0):.4f}")
    
    # 4. Generate LLM risk scores
    logger.info("\n[4/6] Generating LLM risk scores...")
    df['llm_risk_score'] = bert.generate_risk_scores(df)
    df.to_csv('data/synthetic/ghana_credit_data_with_llm_scores.csv', index=False)
    logger.info(f"✓ LLM risk scores added. Mean score: {df['llm_risk_score'].mean():.3f}")
    
    # 5. Train hybrid model
    logger.info("\n[5/6] Training hybrid model...")
    hybrid = HybridCreditModel()
    hybrid_results = hybrid.train(df, include_llm=True)
    hybrid.save('outputs/models/hybrid_model.pkl')
    logger.info(f"✓ Hybrid model trained. AUC: {hybrid_results['auc_roc']:.4f}")
    
    # 6. Calibration and explainability
    logger.info("\n[6/6] Calibration and explainability...")
    
    # Prepare data for calibration
    X = hybrid.prepare_features(df, include_llm=True)
    y = df['default']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_scaled = hybrid.scaler.fit_transform(X_train)
    X_test_scaled = hybrid.scaler.transform(X_test)
    
    # Calibrate
    calibrator = ModelCalibrator()
    calibrated_model = calibrator.calibrate(hybrid.model, X_scaled, y_train)
    calibrator.evaluate_calibration(hybrid.model, X_test_scaled, y_test)
    calibrator.plot_calibration_curve(
        hybrid.model, X_test_scaled, y_test,
        save_path='outputs/figures/calibration_curve.png'
    )
    logger.info("✓ Model calibrated")
    
    # SHAP explainability
    logger.info("Generating SHAP explanations...")
    explainer = ShapExplainer(hybrid.model)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    X_sample = X_test_df.sample(n=min(500, len(X_test_df)), random_state=42)
    explainer.create_explainer(X_sample)
    explainer.plot_summary(X_sample, save_path='outputs/figures/shap_summary.png')
    importance_df = explainer.get_feature_importance()
    importance_df.to_csv('outputs/reports/feature_importance.csv', index=False)
    logger.info("✓ SHAP explanations generated")
    
    # Save results
    results = {
        'baseline_best_model': baseline_name,
        'baseline_auc': baseline_results[baseline_name]['auc_roc'],
        'hybrid_auc': hybrid_results['auc_roc'],
        'improvement': (hybrid_results['auc_roc'] - baseline_results[baseline_name]['auc_roc']) / baseline_results[baseline_name]['auc_roc'] * 100,
        'calibration_brier': calibrator.results['brier_calibrated']
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('outputs/reports/experiment_results.csv', index=False)
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENTS COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
    logger.info(f"\nResults Summary:")
    logger.info(f"  Baseline Model: {baseline_name}")
    logger.info(f"  Baseline AUC: {results['baseline_auc']:.4f}")
    logger.info(f"  Hybrid AUC: {results['hybrid_auc']:.4f}")
    logger.info(f"  Improvement: {results['improvement']:.2f}%")
    logger.info(f"  Calibrated Brier Score: {results['calibration_brier']:.4f}")
    logger.info("\nOutput files:")
    logger.info("  - data/synthetic/ghana_credit_data.csv")
    logger.info("  - data/synthetic/ghana_credit_data_with_llm_scores.csv")
    logger.info("  - outputs/models/ (all trained models)")
    logger.info("  - outputs/figures/ (visualizations)")
    logger.info("  - outputs/reports/ (results and feature importance)")
    logger.info("=" * 70)
    
    return results

if __name__ == "__main__":
    run_all_experiments()