# train_ridge.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import joblib

FEATURE_PATH = "D:/rockfall_ai/data/features/features.csv"
MODEL_PATH = "D:/rockfall_ai/models/ridge_model.joblib"

# Load features
df = pd.read_csv(FEATURE_PATH)
X = df.drop(columns=["normalized_risk"]).values
y = df["normalized_risk"].values

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Save model
joblib.dump(ridge_model, MODEL_PATH)
print(f"✅ Ridge model saved to {MODEL_PATH}")







