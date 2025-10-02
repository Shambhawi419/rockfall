# evaluate.py
import torch
import h5py
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

from train_fusion import (
    UNet, MAX_CHANNELS, IMG_SIZE,
    pad_channels, try_load_dem, compute_slope_aspect,
    synth_weather, synth_sensor, scalar_dict_to_channels
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = r"D:\rockfall_ai\checkpoints\best_multimodal.pt"
TEST_IMG_DIR = Path(r"D:\rockfall_ai\data\drone_img\landslide4sense\TestData\img")
TEST_MASK_DIR = Path(r"D:\rockfall_ai\data\drone_img\landslide4sense\TestData\mask")

# -----------------------------
# Load model
# -----------------------------
model = UNet(in_ch=MAX_CHANNELS).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("Loaded checkpoint for evaluation.")

# DEM / slope / aspect
dem = try_load_dem(None)
slope, aspect = compute_slope_aspect(dem)
slope_t = torch.from_numpy(slope).unsqueeze(0)
aspect_t = torch.from_numpy(aspect).unsqueeze(0)

# -----------------------------
# Metric accumulators
# -----------------------------
y_true_all, y_pred_all = [], []

for img_path in sorted(TEST_IMG_DIR.glob("*.h5")):
    with h5py.File(img_path, "r") as f:
        img = np.array(f["img"], dtype=np.float32)/255.0
    x = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0)

    # Weather/sensor channels
    weather = synth_weather()
    sensor = synth_sensor()
    ws_ch = scalar_dict_to_channels({**weather, **sensor}, IMG_SIZE)

    # Full input
    x_full = torch.cat([x.squeeze(0), slope_t, aspect_t, ws_ch], dim=0)
    x_full = pad_channels(x_full).unsqueeze(0).to(DEVICE)

    # Ground truth mask
    mask_path = TEST_MASK_DIR / f"mask_{img_path.stem.split('_')[-1]}.h5"
    if mask_path.exists():
        with h5py.File(mask_path, "r") as f:
            y_true = np.array(f["mask"], dtype=np.float32)
    else:
        y_true = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Prediction
    with torch.no_grad():
        logits = model(x_full)
        probs = torch.sigmoid(logits).cpu().squeeze().numpy()
        y_pred = (probs > 0.5).astype(np.uint8)

    # Flatten for metrics
    y_true_all.append(y_true.flatten())
    y_pred_all.append(y_pred.flatten())

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

# -----------------------------
# Metrics
# -----------------------------
acc = (y_true_all == y_pred_all).mean()

tp = np.sum((y_true_all == 1) & (y_pred_all == 1))
tn = np.sum((y_true_all == 0) & (y_pred_all == 0))
fp = np.sum((y_true_all == 0) & (y_pred_all == 1))
fn = np.sum((y_true_all == 1) & (y_pred_all == 0))

precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)
iou       = tp / (tp + fp + fn + 1e-8)

# Confusion matrix
cm = confusion_matrix(y_true_all, y_pred_all)
per_class_iou = []
for cls in [0, 1]:
    tp_c = np.sum((y_true_all == cls) & (y_pred_all == cls))
    fp_c = np.sum((y_true_all != cls) & (y_pred_all == cls))
    fn_c = np.sum((y_true_all == cls) & (y_pred_all != cls))
    iou_c = tp_c / (tp_c + fp_c + fn_c + 1e-8)
    per_class_iou.append(iou_c)

# -----------------------------
# Print results
# -----------------------------
print("\n--- Evaluation Results ---")
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"IoU (mean) : {iou:.4f}")

print("\nConfusion Matrix [rows=true, cols=pred]:")
print(cm)

print("\nPer-class IoU:")
print(f"  Background (0): {per_class_iou[0]:.4f}")
print(f"  Landslide  (1): {per_class_iou[1]:.4f}")

