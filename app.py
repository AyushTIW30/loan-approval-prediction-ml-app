import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dict = pickle.load(open("loan_models.pkl", "rb"))

log_model = data_dict["log_model"]
rf_model = data_dict["rf_model"]
log_acc = data_dict["log_acc"]
rf_acc = data_dict["rf_acc"]
features = data_dict["features"]

st.title("💳 Loan Approval Prediction System (ML App)")

model_choice = st.selectbox("Choose ML Model", ["Logistic Regression", "Random Forest"])

st.subheader("Enter Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.selectbox("Self Employed", ["No", "Yes"])
income = st.number_input("Applicant Income", 1000, 100000, step=500)
co_income = st.number_input("Coapplicant Income", 0, 50000, step=500)
loan_amt = st.number_input("Loan Amount", 50, 1000, step=10)
loan_term = st.selectbox("Loan Term (months)", [360, 180, 240, 120])
credit = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

def encode(val, options):
    return options.index(val)

input_data = np.array([[
    encode(gender, ["Female", "Male"]),
    encode(married, ["No", "Yes"]),
    encode(dependents, ["0", "1", "2", "3+"]),
    encode(education, ["Graduate", "Not Graduate"]),
    encode(self_emp, ["No", "Yes"]),
    income,
    co_income,
    loan_amt,
    loan_term,
    credit,
    encode(property_area, ["Rural", "Semiurban", "Urban"])
]])

if model_choice == "Logistic Regression":
    model = log_model
    acc = log_acc
else:
    model = rf_model
    acc = rf_acc

if st.button("Check Loan Status"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.info(f"Model Accuracy: {acc*100:.2f}%")

    st.info(f"Approval Probability: {probability:.2f}%")

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")
        st.snow()

    # Feature importance (only for Random Forest)
    if model_choice == "Random Forest":
        st.subheader("Feature Importance")

        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        ax.barh(fi_df["Feature"], fi_df["Importance"])
        ax.invert_yaxis()
        st.pyplot(fig)
