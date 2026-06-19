"""
CREDIT RISK ASSESSMENT - QUICK START
Daniel Agyekum Amakye - IT University of Copenhagen
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CREDIT RISK ASSESSMENT - QUICK START")
print("Daniel Agyekum Amakye - IT University of Copenhagen")
print("=" * 60)

# ============================================
# STEP 1: Generate Ghana-Specific Synthetic Data
# ============================================

print("\n[1/4] Generating Ghana-specific synthetic data...")

np.random.seed(42)
n_samples = 10000

# Regions with Ghana population distribution
regions = np.random.choice(
    ['Greater Accra', 'Ashanti', 'Western', 'Central', 'Eastern', 'Volta', 'Northern'],
    n_samples,
    p=[0.25, 0.20, 0.12, 0.10, 0.10, 0.08, 0.15]
)

# Demographics
age = np.clip(np.random.normal(34, 11, n_samples).astype(int), 18, 75)
gender = np.random.choice(['Male', 'Female'], n_samples)
employment = np.random.choice(
    ['Full-time', 'Self-employed', 'Part-time', 'Unemployed', 'Farmer', 'Trader'],
    n_samples,
    p=[0.30, 0.25, 0.15, 0.08, 0.12, 0.10]
)

# Income (GHS)
monthly_income = np.random.choice(
    [500, 800, 1200, 1800, 2500, 4000, 6000, 10000],
    n_samples,
    p=[0.08, 0.12, 0.20, 0.25, 0.15, 0.10, 0.06, 0.04]
)

# Credit features
credit_score = np.clip(np.random.normal(600, 100, n_samples).astype(int), 300, 850)
existing_loans = np.clip(np.random.poisson(1.2, n_samples), 0, 8).astype(int)

# Alternative data - Mobile Money
has_mobile_money = np.random.choice([0, 1], n_samples, p=[0.25, 0.75])
mobile_tenure = np.where(has_mobile_money == 1, 
                         np.random.exponential(24, n_samples).astype(int), 0)

# Utility payments
has_utility = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
utility_on_time = np.where(has_utility == 1,
                           np.random.beta(7, 2.5, n_samples), 0)

# Bank account
has_bank_account = np.random.choice([0, 1], n_samples, p=[0.50, 0.50])

# Loan details
loan_amount = np.random.exponential(5000, n_samples).astype(int)
loan_purpose = np.random.choice(
    ['Business', 'Education', 'Medical', 'Home', 'Agriculture', 'Trading'],
    n_samples
)

# Default probability (Ghana NPL ratio ~21%)
default_prob = np.zeros(n_samples)
for i in range(n_samples):
    base = 0.21
    adj = (
        (credit_score[i] - 600) / 100 * -0.08
        + (employment[i] == 'Unemployed') * 0.15
        + (1 - utility_on_time[i]) * 0.1
        + (1 - has_mobile_money[i]) * 0.05
        - has_bank_account[i] * 0.05
    )
    default_prob[i] = np.clip(base + adj, 0.05, 0.50)

default = np.random.binomial(1, default_prob, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'region': regions,
    'employment_status': employment,
    'monthly_income_ghs': monthly_income,
    'credit_score': credit_score,
    'existing_loans': existing_loans,
    'has_mobile_money': has_mobile_money,
    'mobile_money_tenure_months': mobile_tenure,
    'has_utility_account': has_utility,
    'utility_on_time_rate': utility_on_time,
    'has_bank_account': has_bank_account,
    'loan_amount_ghs': loan_amount,
    'loan_purpose': loan_purpose,
    'default': default
})

# Encode categorical variables
df['gender_enc'] = (df['gender'] == 'Male').astype(int)
df['region_enc'] = df['region'].astype('category').cat.codes
df['employment_enc'] = df['employment_status'].astype('category').cat.codes
df['loan_purpose_enc'] = df['loan_purpose'].astype('category').cat.codes

print(f"✅ Generated {len(df):,} records")
print(f"   Default rate: {df['default'].mean():.2%}")
print(f"   Mobile money penetration: {df['has_mobile_money'].mean():.2%}")

# ============================================
# STEP 2: Prepare Features
# ============================================

print("\n[2/4] Preparing features...")

features = [
    'age', 'gender_enc', 'region_enc', 'employment_enc',
    'monthly_income_ghs', 'credit_score', 'existing_loans',
    'has_mobile_money', 'mobile_money_tenure_months',
    'has_utility_account', 'utility_on_time_rate', 'has_bank_account',
    'loan_amount_ghs', 'loan_purpose_enc'
]

X = df[features].copy()
y = df['default'].copy()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE for imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"   Training: {len(X_train):,}")
print(f"   Test: {len(X_test):,}")
print(f"   After SMOTE: {len(X_train_resampled):,}")

# ============================================
# STEP 3: Train Models
# ============================================

print("\n[3/4] Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}

results = {}
best_model = None
best_auc = 0

for name, model in models.items():
    print(f"\n   Training {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {'accuracy': acc, 'auc_roc': auc}
    
    print(f"      Accuracy: {acc:.4f} ({acc:.2%})")
    print(f"      AUC-ROC: {auc:.4f} ({auc:.2%})")
    
    if auc > best_auc:
        best_auc = auc
        best_model = name

print(f"\n✅ Best Model: {best_model} (AUC: {best_auc:.4f})")

# ============================================
# STEP 4: Sample Predictions
# ============================================

print("\n[4/4] Making sample predictions...")

# Get best model
best_model_obj = models[best_model]
y_prob_best = best_model_obj.predict_proba(X_test_scaled)[:, 1]
y_pred_best = best_model_obj.predict(X_test_scaled)

# Show sample
sample_indices = np.random.choice(len(y_test), 5, replace=False)
print("\nSample Loan Decisions:")
print("-" * 60)
print(f"{'Actual':<10} {'Predicted':<12} {'Probability':<12} {'Confidence':<12}")
print("-" * 60)

for idx in sample_indices:
    actual = "DEFAULT" if y_test.iloc[idx] == 1 else "GOOD"
    predicted = "DEFAULT" if y_pred_best[idx] == 1 else "GOOD"
    prob = y_prob_best[idx]
    confidence = 1 - 2 * abs(prob - 0.5)
    print(f"{actual:<10} {predicted:<12} {prob:.2%}      {confidence:.2%}")

# Summary
print("\n" + "=" * 60)
print("✅ QUICK START COMPLETE!")
print("=" * 60)
print(f"\nResults Summary:")
print(f"  Best Model: {best_model}")
print(f"  Accuracy: {results[best_model]['accuracy']:.4f} ({results[best_model]['accuracy']:.2%})")
print(f"  AUC-ROC: {results[best_model]['auc_roc']:.4f} ({results[best_model]['auc_roc']:.2%})")

# Calculate calibration gap
model_confidence = np.mean(1 - 2 * np.abs(y_prob_best - 0.5))
print(f"  Average Confidence: {model_confidence:.2%}")
print(f"\n🔬 Calibration Gap: {(results[best_model]['accuracy'] - model_confidence):.2%}")
print("\nThis is your PhD research baseline!")
print("=" * 60)