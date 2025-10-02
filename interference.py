# interference.py
import torch
import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
TEST_IMG_DIR = Path(r"D:\rockfall_ai\data\drone_img\landslide4sense\TestData\img")
TEST_MASK_DIR = Path(r"D:\rockfall_ai\data\drone_img\landslide4sense\TestData\mask")
SAVE_PRED_DIR = Path(r"D:\rockfall_ai\data\drone_img\landslide4sense\TestData\pred_masks")
SAVE_PRED_DIR.mkdir(exist_ok=True, parents=True)
SAVE_OVERLAY_DIR = SAVE_PRED_DIR / "overlays"
SAVE_OVERLAY_DIR.mkdir(exist_ok=True, parents=True)

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
def save_overlay(x, y_true=None, y_pred=None, save_path=None, alpha=0.4):
    img = x[:3].permute(1,2,0).cpu().numpy()
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(12,4))

    # Original RGB
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("RGB")
    plt.axis("off")

    # Ground truth overlay
    plt.subplot(1,3,2)
    plt.imshow(img)
    if y_true is not None:
        gt_mask = y_true.squeeze().cpu().numpy()
        plt.imshow(gt_mask, cmap='Reds', alpha=alpha)
    plt.title("GT Overlay")
    plt.axis("off")

    # Predicted overlay
    plt.subplot(1,3,3)
    plt.imshow(img)
    if y_pred is not None:
        pred_mask = (y_pred.squeeze().cpu().numpy() > 0.5)
        plt.imshow(pred_mask, cmap='Blues', alpha=alpha)
    plt.title("Pred Overlay")
    plt.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()  # important: prevents popups

# -----------------------------
# Optional: grid overlay visualization
# -----------------------------
def show_grid_overlay(x_list, y_pred_list, alpha=0.4, ncols=3):
    n_images = len(x_list)
    nrows = int(np.ceil(n_images / ncols))
    plt.figure(figsize=(4*ncols, 4*nrows))

    for i, (x, y_pred) in enumerate(zip(x_list, y_pred_list)):
        img = x[:3].permute(1,2,0).cpu().numpy()
        img = np.clip(img, 0, 1)
        pred_mask = (y_pred.squeeze(0).cpu().numpy() > 0.5)

        plt.subplot(nrows, ncols, i+1)
        plt.imshow(img)
        plt.imshow(pred_mask, cmap='Reds', alpha=alpha)
        plt.title(f"Predicted {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# -----------------------------
# Process test images
# -----------------------------
x_list, y_pred_list = [], []

for img_path in sorted(TEST_IMG_DIR.glob("*.h5")):
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
    x_full = pad_channels(x_full)
    x_full = x_full.unsqueeze(0).to(DEVICE)

    # Load ground truth
    mask_path = TEST_MASK_DIR / f"mask_{img_path.stem.split('_')[-1]}.h5"
    if mask_path.exists():
        with h5py.File(mask_path, "r") as f:
            mask = f.get("mask")
            y_true = torch.from_numpy(np.array(mask, dtype=np.float32) if mask is not None else np.zeros((IMG_SIZE, IMG_SIZE))).unsqueeze(0).unsqueeze(0)
    else:
        y_true = torch.zeros((1,1,IMG_SIZE,IMG_SIZE))

    # Inference
    with torch.no_grad():
        logits = model(x_full)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float().cpu()

    print(f"{img_path.name} -> min: {probs.min().item():.6f}, max: {probs.max().item():.6f}")

    # -----------------------------
    # Save overlay & prediction only if not already present
    overlay_path = SAVE_OVERLAY_DIR / f"overlay_{img_path.stem}.png"
    pred_file = SAVE_PRED_DIR / f"pred_{img_path.stem}.npy"

    if not overlay_path.exists() or not pred_file.exists():
        save_overlay(x_full.squeeze(0), y_true, probs, save_path=overlay_path)
        print(f"Saved overlay image: {overlay_path}")
        np.save(pred_file, pred_mask.squeeze(0).numpy())
        print(f"Saved predicted mask: {pred_file}")
    else:
        print(f"Files already exist for {img_path.stem}, skipping save.")

    # Append for grid overlay if needed
    x_list.append(x_full.squeeze(0))
    y_pred_list.append(probs)

# Optional: show grid of all predictions at once
# show_grid_overlay(x_list, y_pred_list, alpha=0.4, ncols=3)




