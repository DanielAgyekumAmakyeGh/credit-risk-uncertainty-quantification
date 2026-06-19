"""
BERT fine-tuning for credit risk assessment.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import logging
import os

logger = logging.getLogger(__name__)

class CreditRiskDataset(Dataset):
    """PyTorch Dataset for credit risk classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertCreditFinetuner:
    """Fine-tune BERT for credit risk assessment."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        batch_size: int = 32,
        epochs: int = 4,
        learning_rate: float = 2e-5,
        device: str = None,
        output_dir: str = "outputs/models/bert_credit_model/"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.tokenizer = None
        self.model = None
        self.results = {}
    
    def initialize(self):
        """Initialize tokenizer and model."""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.model.to(self.device)
        logger.info(f"Model initialized on {self.device}")
        return self
    
    def finetune(self, df: pd.DataFrame, text_col: str = 'loan_description', 
                 target_col: str = 'default') -> Dict:
        """Fine-tune BERT on credit risk data."""
        
        if self.tokenizer is None or self.model is None:
            self.initialize()
        
        # Prepare data
        texts = df[text_col].fillna('').astype(str).values
        labels = df[target_col].values
        
        # Split
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        # Create datasets
        train_dataset = CreditRiskDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = CreditRiskDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = CreditRiskDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training
        logger.info(f"Training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            val_auc = self.evaluate(val_loader)
            logger.info(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Test evaluation
        test_auc = self.evaluate(test_loader)
        logger.info(f"Test AUC: {test_auc:.4f}")
        
        self.results = {
            'val_auc': val_auc,
            'test_auc': test_auc,
            'train_loss': avg_loss
        }
        
        return self.results
    
    def evaluate(self, loader):
        """Evaluate model on a data loader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probabilities = torch.softmax(outputs.logits, dim=1)
                all_preds.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return roc_auc_score(all_labels, all_preds)
    
    def generate_risk_scores(self, df: pd.DataFrame, text_col: str = 'loan_description') -> np.ndarray:
        """Generate LLM risk scores for all loans."""
        
        if self.model is None:
            raise ValueError("Model not initialized. Run finetune() first.")
        
        self.model.eval()
        texts = df[text_col].fillna('').astype(str).values
        risk_scores = []
        
        # Create dataset and loader
        dataset = CreditRiskDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Generating risk scores"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probabilities = torch.softmax(outputs.logits, dim=1)
                risk_scores.extend(probabilities[:, 1].cpu().numpy())
        
        return np.array(risk_scores)
    
    def save_model(self, path: str = None):
        """Save fine-tuned model."""
        if path is None:
            path = self.output_dir
        
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = None):
        """Load fine-tuned model."""
        if path is None:
            path = self.output_dir
        
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")