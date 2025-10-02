# train_xgb.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

FEATURE_PATH = "D:/rockfall_ai/data/features/features.csv"
MODEL_PATH = "D:/rockfall_ai/models/xgb_model.joblib"

# Load features
df = pd.read_csv(FEATURE_PATH)
X = df.drop(columns=["normalized_risk"]).values
y = df["normalized_risk"].values

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, verbosity=1)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# Save model
joblib.dump(xgb_model, MODEL_PATH)
print(f"✅ XGB model saved to {MODEL_PATH}")
