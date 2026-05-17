# ATO Refund Fraud Detection 

## Overview
A machine learning system that detects fraudulent tax refund claims using XGBoost and SHAP explainability. Built to mirror real ATO compliance workflows — the model flags suspicious returns and explains exactly which features drove the decision, giving auditors actionable insights rather than just a black-box score.

## Live Demo
[Try the app here] (https://huggingface.co/spaces/Amogh009/ato-fraud-detection)

Enter a tax return's details and get an instant fraud risk score with a SHAP waterfall explanation.

## Key Results
| Metric | Score |
|---|---|
| ROC-AUC | 0.90 |
| Fraud Recall | 0.81 |
| Fraud Precision | 0.87 |
| Accuracy | 0.98 |

## Fraud Patterns Detected
- Inflated work expense claims relative to income
- Fake travel claims in non-travel industries
- Sudden deduction spikes from prior year
- Income suppression to inflate deduction ratio
- Education expense rorts

## Tech Stack
- **Model:** XGBoost with `scale_pos_weight` for class imbalance
- **Explainability:** SHAP waterfall and summary plots
- **App:** Streamlit deployed on Hugging Face Spaces
- **Data:** 50,000 synthetic tax returns with domain-logic fraud labels

## How to Run Locally
```bash
git clone https://github.com/amoghhatake/ato-fraud-detection
cd ato-fraud-detection
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
├── app.py                  # Streamlit app
├── train.ipynb             # Model training and evaluation
├── EDA.ipynb               # Exploratory data analysis + dataset generation
├── model/                  # Saved model files
└── tax_returns.csv         # Synthetic dataset
```
