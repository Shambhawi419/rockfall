import torch
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

from train_fusion import UNet, MAX_CHANNELS, IMG_SIZE, pad_channels, try_load_dem, compute_slope_aspect, synth_weather, synth_sensor, scalar_dict_to_channels

# -----------------------------
# Paths
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = r"D:\rockfall_ai\checkpoints\best_multimodal.pt"
NEW_IMG_DIR = Path(r"D:\rockfall_ai\data\drone_img\NewData\img")
SAVE_PRED_DIR = Path(r"D:\rockfall_ai\data\drone_img\NewData\pred_masks")
SAVE_OVERLAY_DIR = SAVE_PRED_DIR / "overlays"
SAVE_CSV_DIR = SAVE_PRED_DIR / "csv"

for p in [SAVE_PRED_DIR, SAVE_OVERLAY_DIR, SAVE_CSV_DIR]:
    p.mkdir(exist_ok=True, parents=True)

# -----------------------------
# Load model
# -----------------------------
model = UNet(in_ch=MAX_CHANNELS).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("Loaded checkpoint and model ready for demo inference.")

# -----------------------------
# DEM / slope / aspect
# -----------------------------
dem = try_load_dem(None)
slope, aspect = compute_slope_aspect(dem)
slope_t = torch.from_numpy(slope).unsqueeze(0)
aspect_t = torch.from_numpy(aspect).unsqueeze(0)

# -----------------------------
# Overlay function
# -----------------------------
def overlay_risk(x_full, risk_map, alpha=0.4):
    rgb_img = x_full[:3]
    if isinstance(rgb_img, torch.Tensor):
        rgb_img = rgb_img.cpu().numpy()
    rgb_img = np.transpose(rgb_img, (1,2,0))
    rgb_img = np.clip(rgb_img, 0,1)

    colors = np.array([
        [0,255,0],   # Low risk = green
        [255,255,0], # Medium risk = yellow
        [255,0,0]    # High risk = red
    ], dtype=np.uint8)

    risk_map = risk_map.astype(int)
    color_map = colors[risk_map]
    overlay = (alpha * color_map + (1-alpha) * (rgb_img*255)).astype(np.uint8)
    return overlay

# -----------------------------
# CSV saving function
# -----------------------------
def save_risk_csv(risk_map, csv_dir, image_name):
    H, W = risk_map.shape
    # Per-pixel CSV
    df_pixels = pd.DataFrame({
        "image": image_name,
        "y": np.repeat(np.arange(H), W),
        "x": np.tile(np.arange(W), H),
        "risk_level": risk_map.flatten()
    })
    df_pixels.to_csv(csv_dir / f"{image_name}_pixels.csv", index=False)

    # Summary CSV
    counts = pd.Series(risk_map.flatten()).value_counts()
    df_summary = pd.DataFrame({
        "image": [image_name],
        "low_count": [counts.get(0,0)],
        "medium_count": [counts.get(1,0)],
        "high_count": [counts.get(2,0)]
    })
    df_summary.to_csv(csv_dir / f"{image_name}_summary.csv", index=False)

# -----------------------------
# Demo inference (one image at a time)
# -----------------------------
new_images = sorted(NEW_IMG_DIR.glob("*.h5"))

for img_path in new_images:
    stem = img_path.stem
    overlay_file = SAVE_OVERLAY_DIR / f"overlay_{stem}.png"
    
    # Skip if already processed
    if overlay_file.exists():
        continue

    # Load image
    with h5py.File(img_path,"r") as f:
        img = f.get("img")
        img = np.array(img, dtype=np.float32)/255.0 if img is not None else np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.float32)
    x = torch.from_numpy(img).permute(2,0,1)
    x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE,IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)

    # Weather/Sensor channels
    weather = synth_weather()
    sensor = synth_sensor()
    ws_ch = scalar_dict_to_channels({**weather, **sensor}, IMG_SIZE)

    # Full input
    x_full = torch.cat([x, slope_t, aspect_t, ws_ch], dim=0)
    x_full = pad_channels(x_full).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(x_full)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # Risk map
    risk_map = np.zeros_like(probs, dtype=np.uint8)
    risk_map[probs > 0.7] = 2
    risk_map[(probs > 0.4) & (probs <= 0.7)] = 1
    risk_map[probs <= 0.4] = 0
    risk_map = risk_map.squeeze()

    # Overlay
    overlay = overlay_risk(x_full.squeeze(0).cpu(), risk_map)
    plt.imsave(overlay_file, overlay)

    # Save mask and CSV
    np.save(SAVE_PRED_DIR / f"pred_{stem}.npy", risk_map)
    save_risk_csv(risk_map, SAVE_CSV_DIR, stem)

    # Optional preview for demo (comment out if not needed)
    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

    print(f"Processed {stem}: overlay saved, CSV saved, mask saved.")
    # Process only one image at a time for demo
    break
