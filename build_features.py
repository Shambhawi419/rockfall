# build_features.py
import os
import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from train_fusion import UNet, MAX_CHANNELS, IMG_SIZE, pad_channels, try_load_dem, compute_slope_aspect, synth_weather, synth_sensor, scalar_dict_to_channels

# ---------------- Paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = r"D:\rockfall_ai\checkpoints\best_multimodal.pt"
TRAIN_DIR = Path(r"D:\rockfall_ai\data\drone_img\landslide4sense\TrainData\img")
FEATURE_PATH = Path(r"D:\rockfall_ai\data\features\features.csv")
FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------- Load U-Net
model = UNet(in_ch=MAX_CHANNELS).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("✅ U-Net loaded.")

# ---------------- DEM / slope / aspect
dem = try_load_dem(None)
slope, aspect = compute_slope_aspect(dem)
slope_t = torch.from_numpy(slope).unsqueeze(0)
aspect_t = torch.from_numpy(aspect).unsqueeze(0)

# ---------------- Feature extraction
def extract_features(img_path):
    with h5py.File(img_path, "r") as f:
        img = np.array(f.get("img"), dtype=np.float32)/255.0
    x = torch.from_numpy(img).permute(2,0,1).float()
    x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)
    ws_ch = scalar_dict_to_channels({**synth_weather(), **synth_sensor()}, IMG_SIZE)
    x_full = torch.cat([x, slope_t, aspect_t, ws_ch], dim=0)
    x_full = pad_channels(x_full).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x_full)
        probs = torch.sigmoid(logits)
    feat = probs.flatten().cpu().numpy()
    return feat

# ---------------- Build feature dataset
features_list = []
labels_list = []

for img_file in TRAIN_DIR.glob("*.h5"):
    feat = extract_features(img_file)
    features_list.append(feat)

    # Use mask to compute normalized risk
    with h5py.File(img_file, "r") as f:
        if "mask" in f:
            mask = np.array(f.get("mask"), dtype=np.float32)
            norm_risk = ((mask == 1).sum()*0.5 + (mask > 1).sum()*1.0)/mask.size
        else:
            norm_risk = np.random.rand()  # placeholder
    labels_list.append(norm_risk)

# ---------------- Save to CSV
X = np.array(features_list)
y = np.array(labels_list)
df = pd.DataFrame(X)
df["normalized_risk"] = y
df.to_csv(FEATURE_PATH, index=False)
print(f"✅ Features saved to {FEATURE_PATH}")
