import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved assets
model = joblib.load('churn_rf_model.pkl')
scaler = joblib.load('churn_scaler.pkl')
expected_columns = joblib.load('expected_columns.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# 2. Build the UI
st.title("🚨 Customer Churn Risk Analyzer")
st.write("Enter the customer's profile below to calculate their risk of canceling service.")

st.markdown("### Customer Profile")
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months with company)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=15.0, max_value=8000.0, value=840.0)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

# 3. Process Input and Predict
if st.button("Calculate Churn Risk", type="primary"):
    
    # Put inputs into a DataFrame
    input_dict = {
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract],
        'PaperlessBilling': [paperless],
        'InternetService': [internet_service],
        'PaymentMethod': [payment_method],
        # Fill remaining required categorical fields with safe default assumptions
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No']
    }
    
    input_df = pd.DataFrame(input_dict)
    
    # Apply One-Hot Encoding
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    
    # MAGIC TRICK: Align the new encoded columns with the exact columns the model expects
    # If a column is missing (e.g., the user didn't pick Fiber Optic), it fills it with 0.
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # Scale the numerical columns
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_encoded[numerical_columns] = scaler.transform(input_encoded[numerical_columns])
    
    # Get Probability of Churn (Class 1)
    churn_probability = model.predict_proba(input_encoded)[0][1]
    risk_percentage = churn_probability * 100
    
    # 4. Display the Results
    st.markdown("---")
    st.subheader("Analysis Results")
    
    # Visual Risk Gauge
    if risk_percentage > 50:
        st.error(f"High Risk of Churn: {risk_percentage:.1f}%")
        st.progress(int(risk_percentage))
        st.write("**Recommendation:** High flight risk. Immediately offer a promo code or attempt to lock them into a 1-year contract.")
    elif risk_percentage > 30:
        st.warning(f"Moderate Risk of Churn: {risk_percentage:.1f}%")
        st.progress(int(risk_percentage))
        st.write("**Recommendation:** Monitor this account closely. Ensure they are satisfied with their current tech support.")
    else:
        st.success(f"Low Risk of Churn: {risk_percentage:.1f}%")
        st.progress(int(risk_percentage))
        st.write("**Recommendation:** Healthy account. No immediate action required.")
