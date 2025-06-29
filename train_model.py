import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

print("🚀 Training script started...")

# Base path setup
base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "dataset", "lahore_dataset.csv")
data = pd.read_csv(dataset_path)
print("✅ Dataset loaded.")

# Normalize categorical columns (lowercase and strip)
categorical_cols = ["service_type", "main_category", "area_name", "day_of_week", "demand_level", "weather"]
for col in categorical_cols:
    data[col] = data[col].str.strip().str.lower()

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print("🔠 Categorical columns normalized and encoded.")

# Features and target
X = data.drop(columns=["final_price"])
y = data["final_price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation :")
print(f"MAE  (Mean Absolute Error): {mae:.2f}")
print(f"R²   (R-squared Score): {r2:.2f}")

print("✅ Model trained.")

# Save model and encoders
model_dir = os.path.join(base_dir, "model")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "price_predictor.pkl"))
joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.pkl"))
print("✅ Model and encoders saved in /model")
