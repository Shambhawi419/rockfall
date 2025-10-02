import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

# -------------------- Paths
FEATURE_PATH = Path(r"D:\rockfall_ai\data\features\features.csv")
MODEL_DIR = Path(r"D:\rockfall_ai\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "rf_model.joblib"

# -------------------- Load dataset
print("📂 Loading features...")
df = pd.read_csv(FEATURE_PATH)

X = df.drop(columns=["normalized_risk"]).values
y = df["normalized_risk"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Dataset loaded: {X_train.shape[0]} train, {X_val.shape[0]} val")

# -------------------- Chunked training
total_trees = 100   # total desired estimators
chunk_size = 20     # train 20 trees at a time
n_chunks = total_trees // chunk_size

final_model = None

for i in range(n_chunks):
    print(f"\n🚀 Training chunk {i+1}/{n_chunks} ({chunk_size} trees)...")
    rf = RandomForestRegressor(
        n_estimators=chunk_size,
        max_depth=15,
        random_state=42 + i,
        warm_start=True,
        n_jobs=-1,
        verbose=1
    )
    
    # If we already have a partial model, reuse it (warm start)
    if final_model is not None:
        rf.estimators_ = final_model.estimators_
    
    rf.fit(X_train, y_train)
    
    # Save intermediate model
    joblib.dump(rf, MODEL_PATH)
    final_model = rf
    print(f"💾 Saved intermediate model after {((i+1)*chunk_size)} trees")

print("\n🎉 Training complete! Final model saved at:", MODEL_PATH)
