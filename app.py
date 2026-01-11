import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# 1. Load the Model
@st.cache_resource
def load_data():
    model = pickle.load(open('churn_model.pkl', 'rb'))
    feature_names = pickle.load(open('features.pkl', 'rb'))
    return model, feature_names

model, feature_names = load_data()

# 2. App Interface
st.title("📉 SaaS Churn Predictor")
st.markdown("### Identify at-risk customers instantly")

# 3. Sidebar Inputs
st.sidebar.header("Customer Details")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 10.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 10.0, 8000.0, 1000.0)

# Simulate other inputs (simplified for demo)
# In a real app, you would add dropdowns for every single feature
num_services = st.sidebar.slider("Number of Services", 0, 8, 2)

# 4. Prepare Input Data
input_df = pd.DataFrame(0, index=[0], columns=feature_names)
input_df['tenure'] = tenure
input_df['MonthlyCharges'] = monthly_charges
input_df['TotalCharges'] = total_charges
input_df['Num_Services'] = num_services

# 5. Prediction Logic
if st.button("Predict Risk"):
    prob = model.predict_proba(input_df)[0][1]
    
    st.write("---")
    st.subheader(f"Churn Probability: {prob:.1%}")
    
    if prob > 0.5:
        st.error("⚠️ HIGH RISK: Customer likely to leave.")
    else:
        st.success("✅ LOW RISK: Customer likely to stay.")

    # SHAP Explanation
    st.subheader("Why?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig)
