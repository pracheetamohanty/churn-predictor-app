import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# --- 1. Page Configuration (The "App" Look) ---
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide", # This makes it fill the whole screen
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS (The "Designer" Touch) ---
# This hides standard Streamlit branding and makes headers pop
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div.stButton > button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Load Resources ---
@st.cache_resource
def load_data():
    model = pickle.load(open('churn_model.pkl', 'rb'))
    feature_names = pickle.load(open('features.pkl', 'rb'))
    return model, feature_names

model, feature_names = load_data()

# --- 4. Sidebar (The Input Panel) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4144/4144781.png", width=80)
    st.title("🛡️ ChurnGuard AI")
    st.markdown("Adjust customer profile below:")
    st.write("---")
    
    # Inputs
    tenure = st.slider("📅 Tenure (Months)", 1, 72, 12)
    monthly_charges = st.number_input("💵 Monthly Charges ($)", 10.0, 200.0, 70.0)
    total_charges = st.number_input("💰 Total Charges ($)", 10.0, 8000.0, 1000.0)
    num_services = st.slider("🔧 Number of Services", 0, 8, 2)
    
    st.write("---")
    st.caption("Powered by XGBoost & Python")

# --- 5. Main Dashboard Area ---

# Header Section
st.title("📊 Live Retention Dashboard")
st.markdown("Analyze customer risk in real-time using Machine Learning.")

# Create a "Row" of info cards using Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"**Current Tenure:** {tenure} Months")
with col2:
    st.info(f"**Monthly Bill:** ${monthly_charges:.2f}")
with col3:
    st.info(f"**Total Value:** ${total_charges:.2f}")

st.write("") # Spacer

# --- 6. Prediction Logic & Display ---
# We use a container to make the result stand out
with st.container():
    if st.button("🚀 Run Risk Analysis"):
        
        # Prepare input
        input_df = pd.DataFrame(0, index=[0], columns=feature_names)
        input_df['tenure'] = tenure
        input_df['MonthlyCharges'] = monthly_charges
        input_df['TotalCharges'] = total_charges
        input_df['Num_Services'] = num_services
        
        # Predict
        prob = model.predict_proba(input_df)[0][1]
        
        # --- The "Big Reveal" UI ---
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            # Gauge-like visual using a progress bar
            st.write("### Risk Probability")
            st.metric(label="Churn Risk", value=f"{prob:.1%}", delta=f"{'High' if prob > 0.5 else 'Low'} Risk", delta_color="inverse")
            st.progress(float(prob))
            
            if prob > 0.5:
                st.error("⚠️ ACTION NEEDED: High probability of churn!")
            else:
                st.success("✅ CUSTOMER SAFE: Low probability of churn.")

        with col_res2:
            st.write("### 🔍 Why this prediction?")
            with st.spinner("Calculating SHAP values..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                fig, ax = plt.subplots(figsize=(5, 3)) # Make chart smaller to fit
                shap.summary_plot(shap_values, input_df, plot_type="bar", show=False, color_bar=False)
                # Tweak plot specifically for web view
                plt.xlabel("Impact on Churn Risk")
                st.pyplot(fig)

    else:
        # Placeholder when app first loads
        st.info("👈 Adjust inputs in the sidebar and click 'Run Risk Analysis'")
