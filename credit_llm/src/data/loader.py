"""
Data loader for credit risk datasets.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and validate credit risk datasets."""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_lendingclub(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load Lending Club dataset."""
        if filepath is None:
            filepath = self.data_dir / "raw" / "lendingclub_loan.csv"
        
        if not Path(filepath).exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded Lending Club data: {len(df):,} records")
        return df
    
    def load_ghana_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load Ghana-specific synthetic data."""
        if filepath is None:
            filepath = self.data_dir / "synthetic" / "ghana_credit_data.csv"
        
        if not Path(filepath).exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded Ghana data: {len(df):,} records")
        return df
    
    def load_with_llm_scores(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load data with LLM risk scores."""
        if filepath is None:
            filepath = self.data_dir / "synthetic" / "ghana_credit_data_with_llm_scores.csv"
        
        if not Path(filepath).exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded data with LLM scores: {len(df):,} records")
        return df