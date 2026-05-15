import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load('model/xgb_fraud_model.pkl')
model_columns = joblib.load('model/model_columns.pkl')

import shap
print(shap.__version__)





st.title('ATO Refund Fraud Detection')
st.write('Enter tax return details below to assess the risk')
income = st.number_input("Income Declared (AUD)", min_value=0, value=70000)
work_expense = st.number_input("Work Expense Claim (AUD)", min_value=0, value=10000)
travel_expense = st.number_input("Travel Expense Claim (AUD)", min_value=0, value=2000)
education_expense = st.number_input("Education Expense (AUD)", min_value=0, value=1000)
other_deductions = st.number_input("Other Deductions (AUD)", min_value=0, value=500)
prior_year_income_delta = st.number_input("Prior Year Income Change (e.g. 0.05 = 5% increase)", value=0.04)
prior_year_deduction_delta = st.number_input("Prior Year Deduction Change (e.g. 0.10 = 10% increase)", value=0.03)
filing_method = st.selectbox("Filing Method", ["self", "tax_agent"])
industry = st.selectbox("Industry", ["construction", "healthcare", "education", "retail", "finance", "IT", "transport", "hospitality", "real_estate", "manufacturing"])
num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

if st.button("Assess Fraud Risk"):
    total_deductions = work_expense + travel_expense + education_expense + other_deductions
    deduction_income_ratio = total_deductions / income if income > 0 else 0
    refund_requested = max(0, (income * 0.325) - (total_deductions * 0.325) * 0.95)
    
    
    input_data = pd.DataFrame([{
        'income_declared': income,
        'work_expense_claim': work_expense,
        'travel_expense_claim': travel_expense,
        'education_expense': education_expense,
        'other_deductions': other_deductions,
        'total_deductions': total_deductions,
        'deduction_income_ratio': deduction_income_ratio,
        'prior_year_income_delta': prior_year_income_delta,
        'prior_year_deduction_delta': prior_year_deduction_delta,
        'refund_requested': refund_requested,
        'num_dependents': num_dependents,
        'filing_method': filing_method,
        'industry_code': industry
        
    }])
    
    input_encoded = pd.get_dummies(input_data, columns=['filing_method', 'industry_code'])
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    input_encoded = input_encoded.astype(float)
    
    st.write(input_encoded)

    pred_proba = model.predict_proba(input_encoded)[:, 1]

    fraud_prob = pred_proba[0]

    if fraud_prob >= 0.7:
        risk_level = "HIGH RISK"
        color = "red"
    elif fraud_prob >= 0.4:
        risk_level = "MEDIUM RISK"
        color = "orange"
    else:
        risk_level = "LOW RISK"
        color = "green"

    st.markdown(f"### Fraud Risk Score: `{fraud_prob:.2%}`")
    st.markdown(f"## :{color}[{risk_level}]")
    st.write("---")
    st.write("### Why was this flagged? (SHAP Explanation)")

    explainer = shap.TreeExplainer(model.get_booster())
    shap_values = explainer(input_encoded)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    

    