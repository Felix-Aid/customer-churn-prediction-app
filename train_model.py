import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the original dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Encode 'Contract'
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["Contract"] = df["Contract"].map(contract_map)

# Drop rows with missing or invalid TotalCharges
df = df[df["TotalCharges"] != " "]
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

# Encode target variable
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Select features
features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Contract"]
X = df[features]
y = df["Churn"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1,
    scoring='f1'
)
grid_search.fit(X_train_scaled, y_train)

# Save best model and scaler
model = grid_search.best_estimator_
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Best Parameters:", grid_search.best_params_)
