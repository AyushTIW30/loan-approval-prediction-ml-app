import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/loan.csv")
df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Split
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Accuracy
log_acc = accuracy_score(y_test, log_model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

# Save everything
pickle.dump({
    "log_model": log_model,
    "rf_model": rf_model,
    "log_acc": log_acc,
    "rf_acc": rf_acc,
    "features": X.columns.tolist()
}, open("loan_models.pkl", "wb"))

print("Models trained & saved!")
print("Logistic Accuracy:", log_acc)
print("RandomForest Accuracy:", rf_acc)
