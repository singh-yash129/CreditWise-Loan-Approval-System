import streamlit as st
import joblib
import pandas as pd

# 1. Load the models
# Ensure these files are in the same folder as app.py
logistic_model = joblib.load('logistic_loan_model.pkl')
rf_model = joblib.load('rf_loan_model.pkl')

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant details below to check loan eligibility.")

# 2. Sidebar for Model Selection
st.sidebar.header("Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose Prediction Model", 
    ("Random Forest (Highly Accurate)", "Logistic Regression")
)

# Assign model based on selection
if "Random Forest" in selected_model_name:
    model = rf_model
else:
    model = logistic_model

# 3. Form for 20-Column Input
with st.form("loan_form"):
    st.subheader("Applicant Information")
    
    # Organizing inputs into columns for a better UI
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
        age = st.number_input("Age", min_value=18.0, max_value=100.0, value=30.0)
        gender = st.selectbox("Gender", ("Male", "Female"))
    with row1_col2:
        co_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
        marital = st.selectbox("Marital Status", ("Single", "Married"))
        dependents = st.number_input("Number of Dependents", min_value=0.0, step=1.0, value=0.0)
    with row1_col3:
        education = st.selectbox("Education Level", ("Graduate", "Undergraduate"))
        employment = st.selectbox("Employment Status", ("Salaried", "Self-employed"))
        emp_cat = st.selectbox("Employer Category", ("Government", "Private", "Consultant", "IT"))

    st.divider()
    st.subheader("Financial Details")
    
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        credit_score = st.slider("Credit Score", 300, 900, 700)
        existing_loans = st.number_input("Existing Loans Count", min_value=0.0, step=1.0, value=0.0)
    with row2_col2:
        savings = st.number_input("Savings Balance", min_value=0.0, value=1000.0)
        dti = st.number_input("DTI Ratio (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.3, format="%.2f")
    with row2_col3:
        collateral = st.number_input("Collateral Value", min_value=0.0, value=0.0)
        property_area = st.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

    st.divider()
    st.subheader("Loan Request Details")
    
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        loan_amount = st.number_input("Requested Loan Amount", min_value=0.0, value=20000.0)
        loan_term = st.number_input("Loan Term (Months)", min_value=1.0, value=360.0)
    with row3_col2:
        purpose = st.selectbox("Loan Purpose", ("Home", "Education", "Personal", "Business"))

    # Submit button
    submitted = st.form_submit_button("Check Loan Status")

# 4. Prediction Logic
if submitted:
    # This dictionary MUST match your original columns exactly
    data = {
        'Applicant_ID': 0.0,  # Required by the model but dummy for input
        'Applicant_Income': income,
        'Coapplicant_Income': co_income,
        'Employment_Status': employment,
        'Age': age,
        'Marital_Status': marital,
        'Dependents': dependents,
        'Credit_Score': credit_score,
        'Existing_Loans': existing_loans,
        'DTI_Ratio': dti,
        'Savings': savings,
        'Collateral_Value': collateral,
        'Loan_Amount': loan_amount,
        'Loan_Term': loan_term,
        'Loan_Purpose': purpose,
        'Property_Area': property_area,
        'Education_Level': education,
        'Gender': gender,
        'Employer_Category': emp_cat
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([data])
    
    # Reorder columns to match original training order
    # (Your model expects columns in a specific order)
    column_order = [
        'Applicant_ID', 'Applicant_Income', 'Coapplicant_Income', 'Employment_Status', 
        'Age', 'Marital_Status', 'Dependents', 'Credit_Score', 'Existing_Loans', 
        'DTI_Ratio', 'Savings', 'Collateral_Value', 'Loan_Amount', 'Loan_Term', 
        'Loan_Purpose', 'Property_Area', 'Education_Level', 'Gender', 'Employer_Category'
    ]
    input_df = input_df[column_order]

    try:
        # Perform prediction
        result = model.predict(input_df)
        
        st.divider()
        if result[0] == "Yes":
            st.balloons()
            st.success("✅ Prediction: The loan is LIKELY TO BE APPROVED!")
        else:
            st.error("❌ Prediction: The loan is LIKELY TO BE REJECTED.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")