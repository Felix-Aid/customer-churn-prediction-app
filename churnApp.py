import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
import sklearn
import os


# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Title
st.title("Customer Churn Prediction App")

# Print versions for debugging
st.write(f"scikit-learn version: {sklearn.__version__}")
st.write(f"SHAP version: {shap.__version__}")

# Define paths for model and scaler
model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")



# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define contract mapping (same as training)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}

# Define sample dataset paths
sample_dir = os.path.join(BASE_DIR, "samples")  # Updated to 'samples'
sample_files = {
    "Sample Dataset 1": os.path.join(sample_dir, "sample1.csv"),
    "Sample Dataset 2": os.path.join(sample_dir, "sample2.csv"),
    "Sample Dataset 3": os.path.join(sample_dir, "sample3.csv")
}

# Display sample file paths for debugging
st.write("Sample file paths:", list(sample_files.values()))

# Data source selection
data_source = st.radio(
    "Select data source:",
    ["Upload CSV"] + list(sample_files.keys()),
    help="Choose to upload your own CSV or select a sample dataset."
)

# Load data based on source
df = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    sample_path = sample_files[data_source]
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        st.error(f"Sample file '{sample_path}' not found. Please ensure it exists in the '{sample_dir}' directory.")

# Process data if loaded
if df is not None:
    # Validate required columns
    required_columns = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Contract"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"CSV must contain columns: {required_columns}")
    else:
        # Handle missing or invalid TotalCharges
        df = df[df["TotalCharges"] != " "]
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=required_columns)

        st.write("Data Preview:", df.head())

        # Feature selection and preprocessing
        X = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Contract"]]
        
        # Label encode 'Contract'
        X["Contract"] = X["Contract"].map(contract_map)
        if X["Contract"].isna().any():
            st.error("Invalid 'Contract' values in CSV. Expected: 'Month-to-month', 'One year', 'Two year'.")
        else:
            # Scale features
            X_scaled = scaler.transform(X)

            # Debugging outputs
            st.write(f"X shape: {X.shape}")
            st.write(f"X_scaled shape: {X_scaled.shape}")
            st.write(f"Features: {X.columns.tolist()}")
            st.write(f"Model n_features_in_: {model.n_features_in_}")

            # Predictions
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]

            # Output results
            df["Prediction"] = np.where(predictions == 1, "Churn", "No Churn")
            df["Probability"] = [f"{p:.2f}%" for p in probabilities * 100]
            st.subheader("📊 Prediction Results")
            st.write(df)

            # Download predictions as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            # SHAP Integration
            st.subheader("🔍 SHAP Model Explanation")
            
            # Initialize TreeExplainer for probability output
            explainer = shap.TreeExplainer(model, X_scaled, model_output="probability")
            shap_values = explainer(X_scaled)

            # Debug: Compare outputs
            shap_sum = explainer.expected_value[1] + shap_values[:, :, 1].sum(axis=1)
            st.write("Model output (first 5):", probabilities[:5])
            st.write("SHAP sum (first 5):", shap_sum[:5])

            # SHAP Summary Plot
            st.markdown("### 🔗 Global Feature Importance (SHAP Summary Plot)")
            fig_summary = plt.figure()
            shap.summary_plot(shap_values[:, :, 1], X, show=False)  # Positive class (churn)
            st.pyplot(fig_summary)

            # SHAP Waterfall Plot
            st.markdown("### 🔎 Local Explanation (SHAP Waterfall Plot for a Selected Customer)")
            selected_index = st.slider("Select customer index", 0, len(X)-1, 0)
            fig_waterfall = plt.figure()
            shap.plots.waterfall(shap_values[selected_index, :, 1], show=False)  # Positive class
            st.pyplot(fig_waterfall)
else:
    if data_source == "Upload CSV":
        st.warning("Please upload a CSV file.")
    else:
        st.warning("Selected sample dataset could not be loaded.")
