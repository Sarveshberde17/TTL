import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# -------------------------------
# 1. Load dataset
# -------------------------------
try:
    df = pd.read_csv("Cleaned_data.csv")
except Exception as e:
    raise RuntimeError(f"Error loading dataset: {e}")

print("Dataset loaded successfully")
print("Columns:", df.columns)

# -------------------------------
# 2. Basic cleaning
# -------------------------------
# Drop unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# Define target column
TARGET = "price"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found")

# -------------------------------
# 3. Split features and target
# -------------------------------
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# -------------------------------
# 4. Handle categorical data
# -------------------------------
# Drop high-cardinality columns (important)
drop_cols = ["locality_name", "region_name"]

for col in drop_cols:
    if col in X.columns:
        X = X.drop(col, axis=1)

# Convert remaining categorical columns
X = pd.get_dummies(X, drop_first=True)

print("Processed feature shape:", X.shape)

# -------------------------------
# 5. Train model (controlled)
# -------------------------------
print("Training model...")

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X, y)

# -------------------------------
# 6. Save model + columns
# -------------------------------
try:
    joblib.dump(model, "RandomForestModel.pkl")
    joblib.dump(X.columns.tolist(), "model_columns.pkl")
except Exception as e:
    raise RuntimeError(f"Error saving model: {e}")

print("Model and columns saved successfully")