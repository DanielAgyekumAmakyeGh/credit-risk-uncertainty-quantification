#!/usr/bin/env python
"""
Main entry point for credit risk assessment system.
Usage: python run.py [--generate|--train|--evaluate|--all]
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.ghana_data_generator import GhanaCreditDataGenerator
from src.models.baseline_models import BaselineModels
from src.models.hybrid_model import HybridCreditModel
from src.llm.bert_finetuner import BertCreditFinetuner

def main():
    parser = argparse.ArgumentParser(description="Credit Risk Assessment System")
    parser.add_argument('--generate', action='store_true', help='Generate synthetic data')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples')
    
    args = parser.parse_args()
    
    if args.all or args.generate:
        print("Generating data...")
        generator = GhanaCreditDataGenerator(n_samples=args.n_samples)
        df = generator.generate()
        df.to_csv('data/synthetic/ghana_credit_data.csv', index=False)
        print(f"✓ Data saved to data/synthetic/ghana_credit_data.csv")
    
    if args.all or args.train:
        print("Training models...")
        import pandas as pd
        df = pd.read_csv('data/synthetic/ghana_credit_data.csv')
        
        # Baseline
        baseline = BaselineModels()
        baseline.train_all(df)
        baseline.save_models()
        print("✓ Baseline models trained")
        
        # BERT
        bert = BertCreditFinetuner(epochs=2)
        bert.initialize()
        bert.finetune(df)
        bert.save_model()
        print("✓ BERT model trained")
        
        # Generate LLM scores
        df['llm_risk_score'] = bert.generate_risk_scores(df)
        df.to_csv('data/synthetic/ghana_credit_data_with_llm_scores.csv', index=False)
        
        # Hybrid
        hybrid = HybridCreditModel()
        hybrid.train(df, include_llm=True)
        hybrid.save()
        print("✓ Hybrid model trained")
    
    if args.all or args.evaluate:
        print("Evaluating models...")
        # Evaluation code here
        pass
    
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()