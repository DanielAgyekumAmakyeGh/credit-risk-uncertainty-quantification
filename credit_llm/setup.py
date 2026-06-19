"""
Setup file for credit_risk_assessment package.
"""

from setuptools import setup, find_packages

setup(
    name="credit_risk_assessment",
    version="1.0.0",
    author="Daniel Agyekum Amakye",
    author_email="daniel.amakey@ruc.edu.gh",
    description="LLM-based credit risk assessment with uncertainty quantification",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "lightgbm>=3.3.0",
        "xgboost>=1.7.0",
        "imbalanced-learn>=0.10.0",
        "shap>=0.41.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "joblib>=1.2.0",
        "tqdm>=4.65.0"
    ],
    python_requires=">=3.8",
)