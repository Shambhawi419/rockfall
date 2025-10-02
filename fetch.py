# fetch_gpt_chatbot.py (Quiet Miner-Friendly Version)
import os
import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from pymongo import MongoClient
from openai import OpenAI
import shutil
import joblib
import shap
import sys

from train_fusion import UNet, MAX_CHANNELS, IMG_SIZE, pad_channels, try_load_dem, compute_slope_aspect, synth_weather, synth_sensor, scalar_dict_to_channels

# ----------------------------- Suppress joblib/SHAP prints
class suppress_stdout_stderr:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# ----------------------------- Paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = r"D:\rockfall_ai\checkpoints\best_multimodal.pt"
TEST_DIR = Path(r"D:\rockfall_ai\data\drone_img\landslide4sense\TestData\img")
NEWDATA_DIR = Path(r"D:\rockfall_ai\data\drone_img\NewData\img")
SAVE_PRED_DIR = Path(r"D:\rockfall_ai\data\drone_img\NewData\pred_masks")
SAVE_OVERLAY_DIR = SAVE_PRED_DIR / "overlays"
SAVE_PRED_DIR.mkdir(exist_ok=True, parents=True)
SAVE_OVERLAY_DIR.mkdir(exist_ok=True, parents=True)
NEWDATA_DIR.mkdir(exist_ok=True, parents=True)

# ----------------------------- MongoDB
MONGO_URI = "mongodb+srv://rockfall_user:rock123@cluster0.2qjbtsd.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client["rockfall_ai"]
summary_collection = db["risk_summary"]
pixels_collection = db["risk_pixels"]
history_collection = db["user_history"]
risk_images_collection = db["risk_images"]

# ----------------------------- OpenAI
client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------- Load models
model = UNet(in_ch=MAX_CHANNELS).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

XGB_MODEL_PATH = r"D:\rockfall_ai\models\xgb_model.joblib"
RF_MODEL_PATH  = r"D:\rockfall_ai\models\rf_model.joblib"
xgb_model = joblib.load(XGB_MODEL_PATH)
rf_model  = joblib.load(RF_MODEL_PATH)

explainer_xgb = shap.Explainer(xgb_model)
explainer_rf  = shap.Explainer(rf_model)
feature_names = None

# ----------------------------- DEM / slope / aspect
dem = try_load_dem(None)
slope, aspect = compute_slope_aspect(dem)
slope_t = torch.from_numpy(slope).unsqueeze(0)
aspect_t = torch.from_numpy(aspect).unsqueeze(0)

# ----------------------------- History helpers
def get_user_history(user_id):
    doc = history_collection.find_one({"user_id": user_id})
    return doc["history"] if doc else []

def save_user_history(user_id, history):
    history_collection.update_one(
        {"user_id": user_id},
        {"$set": {"history": history, "last_access": pd.Timestamp.utcnow()}},
        upsert=True
    )

# ----------------------------- Overlay
def save_overlay(x, y_pred=None, save_path=None, alpha=0.4):
    img = x[:3].permute(1,2,0).cpu().numpy()
    img = np.clip(img,0,1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img); plt.title("RGB"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(img)
    if y_pred is not None:
        plt.imshow(y_pred.squeeze().cpu().numpy()>0.5, cmap='Blues', alpha=alpha)
    plt.title("Pred Overlay"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(img); plt.title("Final Overlay"); plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# ----------------------------- ML Feature extraction & risk
def extract_features(pred_mask, slope, aspect, ws_dict, sensor_dict, fixed_size=16384):
    mask_feat = pred_mask.flatten()
    slope_feat = slope.flatten()
    aspect_feat = aspect.flatten()
    ws_feat = np.array(list(ws_dict.values()))
    sensor_feat = np.array(list(sensor_dict.values()))
    features = np.concatenate([mask_feat, slope_feat, aspect_feat, ws_feat, sensor_feat])
    if features.size > fixed_size:
        features = features[:fixed_size]
    elif features.size < fixed_size:
        features = np.pad(features, (0, fixed_size - features.size), mode='constant')
    return features

def compute_ml_risk_and_shap(features):
    global feature_names
    features = features.reshape(1,-1)
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(features.shape[1])]
    with suppress_stdout_stderr():
        xgb_risk = xgb_model.predict(features)[0]
        rf_risk  = rf_model.predict(features)[0]
        shap_xgb = explainer_xgb(features)
        shap_rf  = explainer_rf(features)
    ensemble_risk = (xgb_risk + rf_risk) / 2
    top_xgb_idx = np.argsort(np.abs(shap_xgb.values[0]))[::-1][:3]
    top_rf_idx  = np.argsort(np.abs(shap_rf.values[0]))[::-1][:3]
    shap_summary = {
        "XGB_top_features": {feature_names[i]: float(shap_xgb.values[0][i]) for i in top_xgb_idx},
        "RF_top_features": {feature_names[i]: float(shap_rf.values[0][i]) for i in top_rf_idx}
    }
    return ensemble_risk, shap_summary

# ----------------------------- Miner-friendly SHAP
def interpret_shap(shap_dict, top_n=3):
    contributions = []
    for model_name, feats in shap_dict.items():
        for feat, val in feats.items():
            contributions.append((feat, float(val)))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    sentences = []
    for feat, val in contributions[:top_n]:
        if 'mask' in feat.lower() or 'feat_0' in feat:
            desc = "wet/saturated soil areas detected by drone imagery"
        elif 'slope' in feat.lower():
            desc = "steep slopes"
        elif 'aspect' in feat.lower():
            desc = "terrain facing risk-prone directions"
        elif 'rain' in feat.lower() or 'ws' in feat.lower():
            desc = "recent rainfall or water saturation"
        elif 'sensor' in feat.lower():
            desc = "sensor-detected ground movement"
        else:
            desc = "other local conditions"
        intensity = "increased" if val > 0 else "reduced"
        sentences.append(f"{intensity} risk due to {desc}")
    return sentences

def generate_miner_summary(summary_doc, shap_dict=None):
    low, medium, high = summary_doc['low_count'], summary_doc['medium_count'], summary_doc['high_count']
    total = summary_doc['total_pixels']
    norm_risk = summary_doc['normalized_risk']
    ml_risk = summary_doc.get('ml_risk_score', None)
    zone_desc = []
    if low > 0:
        zone_desc.append(f"Low-risk areas: {low}/{total} pixels, generally stable slopes")
    if medium > 0:
        zone_desc.append(f"Medium-risk areas: {medium}/{total} pixels, watch for loose rocks or moisture")
    if high > 0:
        zone_desc.append(f"High-risk areas: {high}/{total} pixels, immediate attention needed")
    shap_sentences = interpret_shap(shap_dict) if shap_dict else []
    text = f"Image Analysis:\n- " + "\n- ".join(zone_desc) + "\n"
    text += f"- Normalized U-Net risk: {norm_risk:.2f}\n"
    if ml_risk is not None:
        text += f"- ML ensemble risk score: {ml_risk:.2f}\n"
    if shap_sentences:
        text += "\nWhy this risk:\n- " + "\n- ".join(shap_sentences) + "\n"
    if high > 0 or (ml_risk and ml_risk > 0.5):
        text += "\nAdvice: Restrict personnel near high-risk zones, monitor wet slopes, and remove loose rocks. Inspect regularly."
    else:
        text += "\nAdvice: Routine monitoring; slopes appear stable."
    return text

def query_gpt_miner(user_id, summary_doc, shap_dict=None):
    miner_text = generate_miner_summary(summary_doc, shap_dict=shap_dict)
    prompt = f"""
You are a virtual mine assistant. Translate technical ML + U-Net analysis into clear, actionable instructions for miners.
Current analysis:
{miner_text}

Provide a concise, plain-language summary suitable for mine personnel.
"""
    history = get_user_history(user_id)
    messages = [{"role": "system", "content": "You are a virtual mine assistant, speaking in plain language to miners."}]
    messages.extend(history)
    messages.append({"role":"user","content":prompt})
    response = client_ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300
    )
    reply = response.choices[0].message.content
    history.append({"role":"user","content":prompt})
    history.append({"role":"assistant","content":reply})
    save_user_history(user_id, history)
    return reply

# ----------------------------- Process new image
# ----------------------------- Save overlay to MongoDB
# ----------------------------- Save CSVs & Mongo
def save_csvs_to_mongo(image_name, mask, ml_risk=None, shap_dict=None):
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    mask_flat = mask.numpy().flatten()
    h, w = mask.shape[-2:]
    
    pixels = [{"image": image_name, "x": i % w, "y": i // w, "risk_level": int(mask_flat[i])} 
              for i in range(mask_flat.size)]
    df_pixels = pd.DataFrame(pixels)
    
    pixels_collection.delete_many({"image": image_name})
    pixels_collection.insert_many(pixels)
    
    summary_doc = {
        "image": image_name,
        "low_count": int(sum(mask_flat == 0)),
        "medium_count": int(sum(mask_flat == 1)),
        "high_count": int(sum(mask_flat > 1)),
        "total_pixels": len(mask_flat),
        "normalized_risk": float(((sum(mask_flat == 1)*0.5 + sum(mask_flat>1))/len(mask_flat)) 
                                 if len(mask_flat) > 0 else 0),
        "ml_risk_score": float(ml_risk) if ml_risk else None,
        "shap_summary": shap_dict
    }

    summary_collection.update_one({"image": image_name}, {"$set": summary_doc}, upsert=True)
    # Save CSV locally
    pd.DataFrame([summary_doc]).to_csv(SAVE_PRED_DIR / f"{image_name}_summary.csv", index=False)
    
    return summary_doc

# ----------------------------- Save overlay image to MongoDB
def save_image_to_mongo(image_name, overlay_path):
    with open(overlay_path, "rb") as f:
        img_bytes = f.read()
    risk_images_collection.update_one(
        {"image": image_name},
        {"$set": {"overlay": img_bytes}},
        upsert=True
    )

# ----------------------------- Process a new image
def process_new_image(image_name, verbose=True):
    test_path = TEST_DIR / f"{image_name}.h5"
    if not test_path.exists():
        print(f"Image {image_name} not found in TestData folder.")
        return None

    new_path = NEWDATA_DIR / f"{image_name}.h5"
    if not new_path.exists():
        shutil.copy(test_path, new_path)
        if verbose:
            print(f"{image_name} copied to NewData.")

    with h5py.File(new_path, "r") as f:
        img = np.array(f.get("img"), dtype=np.float32)/255.0

    # Convert to tensor and resize
    x = torch.from_numpy(img).permute(2,0,1).float()
    x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                      mode='bilinear', align_corners=False).squeeze(0)

    # Weather + sensor channels
    ws_dict = synth_weather()
    sensor_dict = synth_sensor()
    ws_ch = scalar_dict_to_channels({**ws_dict, **sensor_dict}, IMG_SIZE)

    # Full multimodal input
    x_full = torch.cat([x, slope_t, aspect_t, ws_ch], dim=0)
    x_full = pad_channels(x_full).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(x_full)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float().cpu()

    # Save overlay image locally
    overlay_path = SAVE_OVERLAY_DIR / f"overlay_{image_name}.png"
    save_overlay(x_full.squeeze(0), y_pred=probs, save_path=overlay_path, alpha=0.4)

    # ML features
    if verbose:
        print("Extracting ML features...")
    features = extract_features(pred_mask.numpy(), slope, aspect, ws_dict, sensor_dict)

    if verbose:
        print("Computing ML ensemble risk and SHAP contributions...")
    ml_risk, shap_dict = compute_ml_risk_and_shap(features)

    # Save summary CSV + Mongo
    summary = save_csvs_to_mongo(image_name, pred_mask, ml_risk=ml_risk, shap_dict=shap_dict)

    # Save overlay image to Mongo
    save_image_to_mongo(image_name, overlay_path)

    if verbose:
        print(f"Processing complete for {image_name}")

    return summary


# ----------------------------- Interactive chatbot
if __name__ == "__main__":
    user_id = "Sana"
    print("Rockfall GPT Chatbot. Type 'exit' to quit.\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in ["exit", "quit"]:
            break
        if msg.startswith("image:"):
            img_name = msg.split("image:")[1].strip()
            if not img_name.startswith("image_"):
                img_name = "image_" + img_name
            summary = summary_collection.find_one({"image": img_name})
            if not summary:
                summary = process_new_image(img_name)
            reply = query_gpt_miner(user_id, summary, shap_dict=summary.get("shap_summary"))
        else:
            reply = query_gpt(user_id, msg)
        print(f"{reply}\n")
