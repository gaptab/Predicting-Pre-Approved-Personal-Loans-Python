import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

# -----------------------------
# 1. Dummy Data Generation
# -----------------------------
np.random.seed(42)
num_samples = 1000

df = pd.DataFrame({
    "age": np.random.randint(25, 60, num_samples),
    "credit_score": np.random.randint(300, 850, num_samples),
    "income": np.random.randint(30000, 150000, num_samples),
    "existing_loans": np.random.randint(0, 5, num_samples),
    "monthly_expense": np.random.randint(5000, 50000, num_samples),
    "savings_balance": np.random.randint(1000, 500000, num_samples),
})

# Target variables
df["pre_approved_loan"] = np.random.randint(0, 2, num_samples)  # 1 = Yes, 0 = No
df["loan_amount_offered"] = df["income"] * np.random.uniform(0.5, 2.0, num_samples)  # Loan amount ~ 50% to 200% of income

# Save full dataset
df.to_csv("df.csv", index=False)

# -----------------------------
# 2. Train-Test Split
# -----------------------------
features = ["age", "credit_score", "income", "existing_loans", "monthly_expense", "savings_balance"]

X = df[features]
y_loan = df["pre_approved_loan"]  # Classification target
y_amount = df["loan_amount_offered"]  # Regression target

X_train, X_test, y_loan_train, y_loan_test, y_amount_train, y_amount_test = train_test_split(
    X, y_loan, y_amount, test_size=0.2, random_state=42
)

# Save train and test data
train_data = X_train.copy()
train_data["pre_approved_loan"] = y_loan_train
train_data["loan_amount_offered"] = y_amount_train
train_data.to_csv("train_data.csv", index=False)

test_data = X_test.copy()
test_data["pre_approved_loan"] = y_loan_test
test_data["loan_amount_offered"] = y_amount_test
test_data.to_csv("test_data.csv", index=False)

# -----------------------------
# 3. Data Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4. Model Training
# -----------------------------

# Loan Approval Prediction (Classification)
loan_model = RandomForestClassifier(n_estimators=100, random_state=42)
loan_model.fit(X_train_scaled, y_loan_train)

# Loan Amount Estimation (Regression)
loan_amount_model = LinearRegression()
loan_amount_model.fit(X_train_scaled, y_amount_train)

# -----------------------------
# 5. Model Evaluation
# -----------------------------
loan_pred = loan_model.predict(X_test_scaled)
loan_amount_pred = loan_amount_model.predict(X_test_scaled)

print(f"Loan Approval Model Accuracy: {accuracy_score(y_loan_test, loan_pred):.2f}")
print(f"Loan Amount Estimation MAE: {mean_absolute_error(y_amount_test, loan_amount_pred):.2f}")

# -----------------------------
# 6. Predicting for a New Customer
# -----------------------------
new_customer = pd.DataFrame({
    "age": [35],
    "credit_score": [750],
    "income": [85000],
    "existing_loans": [2],
    "monthly_expense": [20000],
    "savings_balance": [150000],
})

# Save new customer data
new_customer.to_csv("new_customer.csv", index=False)

# Apply same scaling
new_customer_scaled = scaler.transform(new_customer)

# Make Predictions
loan_eligibility = loan_model.predict(new_customer_scaled)[0]
predicted_loan_amount = loan_amount_model.predict(new_customer_scaled)[0]

# -----------------------------
# 7. Display Results
# -----------------------------
print("\nNew Customer Predictions:")
print(f"Eligible for Pre-Approved Loan: {'Yes' if loan_eligibility else 'No'}")
print(f"Recommended Loan Amount: ${predicted_loan_amount:.2f}")
