import torch
import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from train_fusion import (
    UNet, MAX_CHANNELS, IMG_SIZE,
    pad_channels, try_load_dem, compute_slope_aspect,
    synth_weather, synth_sensor, scalar_dict_to_channels
)

# -----------------------------
# Paths
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = r"D:\rockfall_ai\checkpoints\best_multimodal.pt"
NEW_IMG_DIR = Path(r"D:\rockfall_ai\data\drone_img\NewData\img")
SAVE_PRED_DIR = Path(r"D:\rockfall_ai\data\drone_img\NewData\pred_masks")
SAVE_PRED_DIR.mkdir(exist_ok=True, parents=True)
SAVE_OVERLAY_DIR = SAVE_PRED_DIR / "overlays"
SAVE_OVERLAY_DIR.mkdir(exist_ok=True, parents=True)
CSV_DIR = SAVE_PRED_DIR / "csv"
CSV_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
model = UNet(in_ch=MAX_CHANNELS).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("Loaded checkpoint and model ready for inference.")

# -----------------------------
# DEM / slope / aspect
# -----------------------------
dem = try_load_dem(None)
slope, aspect = compute_slope_aspect(dem)
slope_t = torch.from_numpy(slope).unsqueeze(0)
aspect_t = torch.from_numpy(aspect).unsqueeze(0)

# -----------------------------
# Overlay save function
# -----------------------------
def save_overlay(x, y_pred=None, save_path=None, alpha=0.4):
    img = x[:3].permute(1,2,0).cpu().numpy()
    img = np.clip(img, 0, 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.imshow(img)
    if y_pred is not None:
        pred_mask = (y_pred.squeeze().cpu().numpy() > 0.5)
        plt.imshow(pred_mask, cmap='Blues', alpha=alpha)
    plt.title("Pred Overlay"); plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(img); plt.title("RGB"); plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# -----------------------------
# Process new images
# -----------------------------
processed_files = {f.stem.replace("pred_","") for f in SAVE_PRED_DIR.glob("pred_*.npy")}

for img_path in sorted(NEW_IMG_DIR.glob("*.h5")):
    stem = img_path.stem
    if stem in processed_files:
        continue  # skip already processed

    # Load image
    with h5py.File(img_path, "r") as f:
        img = f.get("img")
        img = np.array(img, dtype=np.float32)/255.0 if img is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    x = torch.from_numpy(img).permute(2,0,1).float()
    x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)

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
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float().cpu()

    # Save overlay and prediction
    overlay_path = SAVE_OVERLAY_DIR / f"overlay_{stem}.png"
    save_overlay(x_full.squeeze(0), y_pred=probs, save_path=overlay_path)
    pred_file = SAVE_PRED_DIR / f"pred_{stem}.npy"
    np.save(pred_file, pred_mask.squeeze(0).numpy())

    print(f"Processed: {stem}, overlay saved: {overlay_path}, mask saved: {pred_file}")

    # -----------------------------
    # Generate CSVs for watchdog
    # -----------------------------
    # Pixels CSV
    pixel_csv = CSV_DIR / f"{stem}_pixels.csv"
    coords = np.argwhere(pred_mask.squeeze(0).numpy() >= 0)  # every pixel
    risk_levels = (pred_mask.squeeze(0).numpy() > 0.5).astype(int).flatten()
    df_pixels = pd.DataFrame({
        "image": stem,
        "x": coords[:,1],
        "y": coords[:,0],
        "risk_level": risk_levels[:len(coords)]
    })
    df_pixels.to_csv(pixel_csv, index=False)

    # Summary CSV
    low = np.sum(risk_levels == 0)
    medium = np.sum(risk_levels == 1)
    high = np.sum(risk_levels == 2)
    total = low + medium + high
    normalized_risk = ((medium * 0.5) + high * 1.0)/total if total>0 else 0
    df_summary = pd.DataFrame([{
        "image": stem,
        "low_count": low,
        "medium_count": medium,
        "high_count": high,
        "total_pixels": total,
        "normalized_risk": normalized_risk
    }])
    summary_csv = CSV_DIR / f"{stem}_summary.csv"
    df_summary.to_csv(summary_csv, index=False)

print("✅ Incremental inference complete, CSVs generated for watchdog.")
