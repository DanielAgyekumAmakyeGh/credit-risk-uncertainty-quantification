"""
Ghana-specific synthetic data generator.
"""

import pandas as pd
import numpy as np
import random
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GhanaDemographics:
    """Ghana-specific demographic configuration."""
    
    regions: Dict[str, float] = None
    employment_types: Dict[str, float] = None
    income_brackets: Dict[str, Dict] = None
    mobile_money_penetration: Dict[str, float] = None
    loan_purposes: Dict[str, float] = None
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = {
                'Greater Accra': 0.25, 'Ashanti': 0.20, 'Western': 0.10,
                'Central': 0.09, 'Eastern': 0.09, 'Volta': 0.08,
                'Northern': 0.07, 'Bono East': 0.04, 'Upper East': 0.03,
                'Upper West': 0.02, 'Ahafo': 0.01, 'Savannah': 0.01,
                'North East': 0.01, 'Oti': 0.01, 'Western North': 0.01,
                'Bono': 0.01
            }
        
        if self.employment_types is None:
            self.employment_types = {
                'Formal (Public Sector)': 0.12,
                'Formal (Private Sector)': 0.15,
                'Self-employed (Informal)': 0.40,
                'Self-employed (Formal Business)': 0.08,
                'Farmer': 0.10,
                'Trader/Market Vendor': 0.10,
                'Unemployed - Seeking': 0.03,
                'Unemployed - Not Seeking': 0.01,
                'Student': 0.01
            }
        
        if self.income_brackets is None:
            self.income_brackets = {
                'Low': {'range': (300, 1000), 'pct': 0.25},
                'Lower-Middle': {'range': (1001, 2000), 'pct': 0.30},
                'Middle': {'range': (2001, 4000), 'pct': 0.25},
                'Upper-Middle': {'range': (4001, 8000), 'pct': 0.12},
                'High': {'range': (8001, 20000), 'pct': 0.06},
                'Very High': {'range': (20001, 100000), 'pct': 0.02}
            }
        
        if self.mobile_money_penetration is None:
            self.mobile_money_penetration = {
                'Greater Accra': 0.85, 'Ashanti': 0.80, 'Western': 0.75,
                'Central': 0.78, 'Eastern': 0.76, 'Volta': 0.72,
                'Northern': 0.65, 'Bono East': 0.68, 'Upper East': 0.60,
                'Upper West': 0.58, 'Ahafo': 0.65, 'Savannah': 0.55,
                'North East': 0.55, 'Oti': 0.62, 'Western North': 0.70,
                'Bono': 0.68
            }
        
        if self.loan_purposes is None:
            self.loan_purposes = {
                'Business Expansion (SME)': 0.25,
                'Education Fees (School/University)': 0.15,
                'Medical Emergency': 0.12,
                'Agricultural Inputs (Farming)': 0.08,
                'Trading Capital (Market)': 0.12,
                'Home Construction/Repair': 0.10,
                'Vehicle Purchase (Trotro/Taxi)': 0.05,
                'Rent Payment': 0.05,
                'Funeral Expenses': 0.03,
                'Wedding Expenses': 0.03,
                'SME Equipment Purchase': 0.02
            }

class GhanaCreditDataGenerator:
    """Generate Ghana-specific synthetic credit data."""
    
    def __init__(self, n_samples: int = 100000, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        self.demographics = GhanaDemographics()
    
    def generate(self) -> pd.DataFrame:
        """Generate complete Ghana-specific credit dataset."""
        
        logger.info(f"Generating {self.n_samples:,} Ghana-specific records...")
        
        applicant_ids = [f'GH{str(i).zfill(8)}' for i in range(self.n_samples)]
        
        # Demographics
        regions = np.random.choice(
            list(self.demographics.regions.keys()),
            self.n_samples,
            p=list(self.demographics.regions.values())
        )
        
        age = np.clip(np.random.normal(34, 11, self.n_samples).astype(int), 18, 80)
        gender = np.random.choice(['Male', 'Female'], self.n_samples, p=[0.49, 0.51])
        
        employment = np.random.choice(
            list(self.demographics.employment_types.keys()),
            self.n_samples,
            p=list(self.demographics.employment_types.values())
        )
        
        # Income
        monthly_income = np.zeros(self.n_samples)
        for bracket, config in self.demographics.income_brackets.items():
            mask = (np.random.random(self.n_samples) < config['pct']) & (monthly_income == 0)
            low, high = config['range']
            monthly_income[mask] = np.random.randint(low, high, mask.sum())
        
        # Adjust income by employment
        farmer_mask = employment == 'Farmer'
        monthly_income[farmer_mask] = np.random.randint(800, 3000, farmer_mask.sum())
        
        trader_mask = employment == 'Trader/Market Vendor'
        monthly_income[trader_mask] = np.random.randint(1000, 5000, trader_mask.sum())
        
        unemployed_mask = np.array(['Unemployed' in e for e in employment])
        monthly_income[unemployed_mask] = np.random.randint(200, 800, unemployed_mask.sum())
        
        # Traditional credit
        has_credit_history = np.random.choice([0, 1], self.n_samples, p=[0.60, 0.40])
        
        credit_score = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            if has_credit_history[i]:
                credit_score[i] = np.clip(np.random.normal(600, 100), 300, 850)
            else:
                credit_score[i] = np.random.randint(350, 550)
        credit_score = credit_score.astype(int)
        
        existing_loans = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            if has_credit_history[i]:
                existing_loans[i] = np.random.poisson(1.2)
            else:
                existing_loans[i] = np.random.poisson(0.3)
        existing_loans = np.clip(existing_loans, 0, 8).astype(int)
        
        # Mobile money
        region_penetration = [self.demographics.mobile_money_penetration[r] for r in regions]
        has_mobile_money = np.random.random(self.n_samples) < region_penetration
        
        mobile_money_tenure = np.zeros(self.n_samples)
        mobile_money_avg_balance = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            if has_mobile_money[i]:
                mobile_money_tenure[i] = np.random.exponential(24)
                base_balance = monthly_income[i] * 0.3
                mobile_money_avg_balance[i] = np.random.exponential(base_balance)
        
        mobile_money_tenure = np.clip(mobile_money_tenure, 1, 72).astype(int)
        mobile_money_avg_balance = np.clip(mobile_money_avg_balance, 10, 10000).astype(int)
        
        # Utility
        has_utility = np.random.random(self.n_samples) < 0.65
        utility_on_time = np.zeros(self.n_samples)
        utility_tenure = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            if has_utility[i]:
                base_on_time = 0.7 + (monthly_income[i] / 20000) * 0.2
                utility_on_time[i] = np.clip(np.random.normal(base_on_time, 0.1), 0, 1)
                utility_tenure[i] = np.random.exponential(36)
        utility_on_time = np.round(utility_on_time, 2)
        utility_tenure = np.clip(utility_tenure, 1, 120).astype(int)
        
        # Banking
        uses_digital_payments = np.random.random(self.n_samples) < 0.55
        has_bank_account = np.random.random(self.n_samples) < 0.45
        
        # Loan details
        loan_amount = np.zeros(self.n_samples)
        loan_purpose = np.random.choice(
            list(self.demographics.loan_purposes.keys()),
            self.n_samples,
            p=list(self.demographics.loan_purposes.values())
        )
        
        for i in range(self.n_samples):
            base_amount = monthly_income[i] * 3
            if 'Business' in loan_purpose[i] or 'Trading' in loan_purpose[i]:
                multiplier = np.random.uniform(5, 15)
            elif 'Education' in loan_purpose[i]:
                multiplier = np.random.uniform(2, 8)
            else:
                multiplier = np.random.uniform(1, 5)
            loan_amount[i] = base_amount * multiplier
        
        loan_amount = np.clip(loan_amount, 200, 100000).astype(int)
        loan_term = np.random.choice([3, 6, 12, 18, 24, 36], self.n_samples)
        
        # Default probability
        default_prob = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            base_default = 0.21
            adjustments = (
                (credit_score[i] - 600) / 100 * -0.08
                + (employment[i] == 'Unemployed - Seeking') * 0.15
                + (employment[i] == 'Trader/Market Vendor') * 0.05
                + (existing_loans[i] * 0.03)
                + (1 - utility_on_time[i]) * 0.1 if has_utility[i] else 0
                + (1 - has_mobile_money[i]) * 0.05
                - (has_bank_account[i]) * 0.05
            )
            default_prob[i] = np.clip(base_default + adjustments, 0.05, 0.50)
        
        default = np.random.binomial(1, default_prob, self.n_samples)
        
        # Create dataframe
        df = pd.DataFrame({
            'applicant_id': applicant_ids,
            'age': age,
            'gender': gender,
            'region': regions,
            'employment_status': employment,
            'monthly_income_ghs': monthly_income,
            'has_credit_history': has_credit_history.astype(int),
            'credit_score': credit_score,
            'existing_loans': existing_loans,
            'has_mobile_money': has_mobile_money.astype(int),
            'mobile_money_tenure_months': mobile_money_tenure,
            'mobile_money_avg_balance_ghs': mobile_money_avg_balance,
            'has_utility_account': has_utility.astype(int),
            'utility_on_time_rate': utility_on_time,
            'utility_tenure_months': utility_tenure,
            'uses_digital_payments': uses_digital_payments.astype(int),
            'has_bank_account': has_bank_account.astype(int),
            'loan_amount_ghs': loan_amount,
            'loan_term_months': loan_term,
            'loan_purpose': loan_purpose,
            'default': default
        })
        
        # Generate text
        df['loan_description'] = df.apply(self._generate_loan_description, axis=1)
        df['bureau_narrative'] = df.apply(self._generate_bureau_narrative, axis=1)
        
        logger.info(f"Generated {len(df):,} records with {len(df.columns)} features")
        logger.info(f"Default rate: {df['default'].mean():.2%}")
        logger.info(f"Mobile money penetration: {df['has_mobile_money'].mean():.2%}")
        
        return df
    
    def _generate_loan_description(self, row: pd.Series) -> str:
        purpose = row['loan_purpose']
        income = row['monthly_income_ghs']
        region = row['region']
        
        phrases = {
            'Business Expansion (SME)': [
                f"I need capital to expand my small business in {region}. I sell at the local market and have regular customers. My monthly income is GHS {income:,}.",
                f"Operating in {region}, I want to grow my business. My monthly income is GHS {income:,}."
            ],
            'Education Fees': [
                f"My children are in school in {region} and I need help with school fees. My monthly income is GHS {income:,}.",
                f"I am a hardworking parent in {region} trying to pay for education. My monthly income is GHS {income:,}."
            ],
            'Agricultural Inputs': [
                f"As a farmer in {region}, I need funds for seeds and fertilizer. My monthly income is GHS {income:,}.",
                f"The harvest season is approaching. I need inputs for my farm in {region}. My monthly income is GHS {income:,}."
            ],
            'Trading Capital': [
                f"I am a market trader in {region}. I need capital to buy more goods. My monthly income is GHS {income:,}.",
                f"Operating my stall in {region} market requires regular restocking. My monthly income is GHS {income:,}."
            ]
        }
        
        if purpose in phrases:
            return random.choice(phrases[purpose])
        
        return f"I live in {region} and work hard to support my family. I need this loan for {purpose.lower()}. My monthly income is GHS {income:,}. I am committed to repaying."
    
    def _generate_bureau_narrative(self, row: pd.Series) -> str:
        parts = []
        
        if row['has_mobile_money']:
            if row['mobile_money_tenure_months'] > 24:
                parts.append(f"Mobile money user for {row['mobile_money_tenure_months']} months. Positive history with average balance GHS {row['mobile_money_avg_balance_ghs']:,}.")
            else:
                parts.append(f"Recent mobile money adopter ({row['mobile_money_tenure_months']} months).")
        else:
            parts.append("No mobile money account on record.")
        
        if row['has_utility_account']:
            if row['utility_on_time_rate'] > 0.9:
                parts.append("Excellent utility payment record.")
            elif row['utility_on_time_rate'] > 0.7:
                parts.append("Satisfactory utility payment history.")
            else:
                parts.append("Utility payments show irregular patterns.")
        
        if row['has_credit_history']:
            if row['credit_score'] > 700:
                parts.append("Good credit history.")
            else:
                parts.append("Limited or challenged credit history.")
        else:
            parts.append("No formal credit history. Assessment relies on alternative data.")
        
        return " ".join(parts)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generator = GhanaCreditDataGenerator(n_samples=100000)
    df = generator.generate()
    df.to_csv('data/synthetic/ghana_credit_data.csv', index=False)
    print("✓ Data saved to data/synthetic/ghana_credit_data.csv")