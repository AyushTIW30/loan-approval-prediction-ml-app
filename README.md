# 💳 Loan Approval Prediction System (Machine Learning Web App)

An end-to-end Machine Learning project that predicts whether a loan will be approved or rejected based on applicant financial and demographic details.
The trained ML models are deployed using Streamlit to provide an interactive web interface.

---

## Live Link 
https://loan-approval-prediction-ml-app.streamlit.app/

---

## 🚀 Features

- Predicts loan approval using Machine Learning
- Supports two ML models:
  - Logistic Regression
  - Random Forest Classifier
- Displays:
  - Model accuracy
  - Approval probability percentage
- Visualizes Feature Importance (for Random Forest)
- User-friendly Streamlit web interface

---

## 🧠 Machine Learning Workflow

1. Data Cleaning & Handling Missing Values
2. Encoding Categorical Features
3. Train-Test Split
4. Model Training:
   - Logistic Regression
   - Random Forest
5. Model Evaluation using Accuracy Score
6. Model Deployment using Streamlit

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib
- Streamlit

---

## 📂 Project Structure
loan_project/
│
├── data/
│ └── loan.csv
├── model_train.py
├── loan_models.pkl
├── app.py
├── requirements.txt
└── README.md
