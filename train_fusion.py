# train_fusion_full.py
# Laptop-friendly Multimodal UNet for Landslide4Sense
# Full version with Train/Validation/Test, empty mask handling, evaluation, and visualization

import os
import time
import random
from pathlib import Path
import re
import json

import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# -----------------------------
# Config from JSON
# -----------------------------
with open("config.json", "r") as f:
    cfg = json.load(f)

SEED = cfg["SEED"]
DATA_DIR = cfg["DATA_DIR"]
DEM_TIF = cfg["DEM_TIF"]
CHECKPOINT_DIR = cfg["CHECKPOINT_DIR"]
BEST_PATH = cfg["BEST_PATH"]
LAST_PATH = cfg["LAST_PATH"]
IMG_SIZE = cfg["IMG_SIZE"]
BATCH_SIZE = cfg["BATCH_SIZE"]
LR = cfg["LR"]
EPOCHS = cfg["EPOCHS"]
ACCUM_STEPS = cfg["ACCUM_STEPS"]
EARLY_STOP_PATIENCE = cfg["EARLY_STOP_PATIENCE"]
MAX_CHANNELS = cfg["MAX_CHANNELS"]  # pad/truncate to this

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def dice_coeff(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return ((2*inter + eps)/(denom + eps)).mean()

def iou_score(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    return ((inter + eps)/(union + eps)).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice = 1 - dice_coeff(probs, targets)
        return bce + dice

# -----------------------------
# UNet building blocks
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=MAX_CHANNELS, out_ch=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.down4 = Down(base*8, base*8)

        self.up1 = Up(base*8, base*8, base*4)
        self.up2 = Up(base*4, base*4, base*2)
        self.up3 = Up(base*2, base*2, base)
        self.up4 = Up(base, base, base)

        self.outc = nn.Conv2d(base, out_ch, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# -----------------------------
# DEM / Sensor / Weather
# -----------------------------
def try_load_dem(dem_tif_path):
    try:
        import rasterio
        with rasterio.open(dem_tif_path) as src:
            dem = src.read(1).astype(np.float32)
        return dem
    except:
        return None

def compute_slope_aspect(dem):
    if dem is None:
        h, w = IMG_SIZE, IMG_SIZE
        y = np.linspace(-1,1,h)[:, None]
        x = np.linspace(-1,1,w)[None,:]
        base = 1000 - 300*np.exp(-((x**2 + y**2)*3))
        noise = np.random.RandomState(SEED).normal(0,5,(h,w))
        dem = (base + noise).astype(np.float32)
    dy, dx = np.gradient(dem.astype(np.float32))
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180/np.pi
    aspect = np.arctan2(-dx, dy) * 180/np.pi
    return np.nan_to_num(slope).astype(np.float32), np.nan_to_num(aspect).astype(np.float32)

def synth_weather():
    return {"temp": 20 + np.random.randn()*2,
            "humidity": np.clip(70+np.random.randn()*10,30,100),
            "pressure":1010+np.random.randn()*5,
            "wind": np.clip(2+abs(np.random.randn()),0,12),
            "rain": max(0,np.random.randn()*0.5)}

def synth_sensor():
    return {"pore_pressure": 30+np.random.randn()*3,
            "displacement": max(0,abs(np.random.randn()*0.5)),
            "vibration": max(0,abs(np.random.randn()*0.2))}

def scalar_dict_to_channels(d, out_size):
    chans = []
    for k in sorted(d.keys()):
        m = torch.full((1,out_size,out_size), float(d[k]), dtype=torch.float32)
        chans.append(m)
    if len(chans)==0:
        return torch.zeros((0,out_size,out_size), dtype=torch.float32)
    return torch.cat(chans, dim=0)

def pad_channels(x, max_ch=MAX_CHANNELS):
    C,H,W = x.shape
    if C < max_ch:
        pad = torch.zeros((max_ch-C,H,W), dtype=x.dtype)
        x = torch.cat([x,pad], dim=0)
    elif C > max_ch:
        x = x[:max_ch]
    return x

# -----------------------------
# Dataset
# -----------------------------
def discover_pairs(img_folder, mask_folder=None, ext=".h5"):
    """Match images and masks by number in filename"""
    img_folder = Path(img_folder)
    mask_folder = Path(mask_folder) if mask_folder else None
    img_files = sorted(img_folder.glob(f"*{ext}"))
    mask_files = sorted(mask_folder.glob(f"*{ext}")) if mask_folder else []

    def extract_number(p):
        match = re.search(r'\d+', p.stem)
        return int(match.group()) if match else -1

    mask_dict = {extract_number(p): p for p in mask_files}
    pairs = []
    for img in img_files:
        num = extract_number(img)
        mask = mask_dict.get(num, None)
        pairs.append({"image": img, "mask": mask})
    return pairs

class MultimodalDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, img_size=128, augment=False):
        self.pairs = discover_pairs(img_dir, mask_dir)
        self.img_size = img_size
        self.augment = augment
        self.dem = try_load_dem(DEM_TIF)
        self.slope, self.aspect = compute_slope_aspect(self.dem)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img_path = pair["image"]
        mask_path = pair["mask"]

        # Load image
        try:
            with h5py.File(img_path,"r") as f:
                img = np.array(f.get("img", np.zeros((self.img_size,self.img_size,3),dtype=np.float32)), dtype=np.float32)/255.0
        except:
            img = np.zeros((self.img_size,self.img_size,3),dtype=np.float32)
        if img.ndim == 2:
            img = img[...,None]
        x = torch.from_numpy(img).permute(2,0,1)
        x = F.interpolate(x.unsqueeze(0), size=(self.img_size,self.img_size), mode='bilinear', align_corners=False).squeeze(0)

        # Load mask
        if mask_path and mask_path.exists():
            with h5py.File(mask_path,"r") as f:
                mask = np.array(f.get("mask", np.zeros((self.img_size,self.img_size),dtype=np.float32)), dtype=np.float32)
            mask = (mask>0).astype(np.float32)
            y = torch.from_numpy(mask).unsqueeze(0)
            y = F.interpolate(y.unsqueeze(0), size=(self.img_size,self.img_size), mode='nearest').squeeze(0)
        else:
            y = torch.zeros((1,self.img_size,self.img_size), dtype=torch.float32)

        # DEM channels
        slope_t = torch.from_numpy(self.slope).unsqueeze(0)
        aspect_t = torch.from_numpy(self.aspect).unsqueeze(0)
        slope_t = F.interpolate(slope_t.unsqueeze(0), size=(self.img_size,self.img_size), mode='bilinear', align_corners=False).squeeze(0)
        aspect_t = F.interpolate(aspect_t.unsqueeze(0), size=(self.img_size,self.img_size), mode='bilinear', align_corners=False).squeeze(0)

        # Weather/Sensor
        weather = synth_weather()
        sensor = synth_sensor()
        ws_ch = scalar_dict_to_channels({**weather,**sensor}, self.img_size)

        # Concatenate
        x = torch.cat([x, slope_t, aspect_t, ws_ch], dim=0)
        x = pad_channels(x)

        # Augmentation
        if self.augment and torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])

        return x, y

# -----------------------------
# Build loaders
# -----------------------------
def build_loaders(img_size=IMG_SIZE, val_frac=0.1, test_frac=0.1):
    train_dir = os.path.join(DATA_DIR, "TrainData", "img")
    mask_dir = os.path.join(DATA_DIR, "TrainData", "mask")

    full_ds = MultimodalDataset(train_dir, mask_dir, img_size, augment=True)

    # Filter out empty masks
    non_empty_indices = [i for i, pair in enumerate(full_ds.pairs)
                         if pair["mask"] and h5py.File(pair["mask"],'r')["mask"][()].sum()>0]
    if len(non_empty_indices)==0:
        raise ValueError("No non-empty masks found in training dataset!")

    ds_filtered = Subset(full_ds, non_empty_indices)

    # Train/Val/Test split
    indices = list(range(len(ds_filtered)))
    train_idx, temp_idx = train_test_split(indices, test_size=val_frac+test_frac, random_state=SEED)
    val_relative = val_frac/(val_frac+test_frac)
    val_idx, test_idx = train_test_split(temp_idx, test_size=1-val_relative, random_state=SEED)

    train_loader = DataLoader(Subset(ds_filtered, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(ds_filtered, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(ds_filtered, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    in_ch = MAX_CHANNELS
    return train_loader, val_loader, test_loader, in_ch

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device):
    model.eval()
    dice = 0
    iou = 0
    n = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            dice += dice_coeff(probs,y).item()*x.size(0)
            iou += iou_score(probs,y).item()*x.size(0)
            n += x.size(0)
    return dice/n, iou/n

def show_predictions(x, y_true, y_pred, n=3):
    x = x.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    for i in range(min(n, x.shape[0])):
        plt.figure(figsize=(8,3))
        plt.subplot(1,3,1)
        plt.imshow(np.transpose(x[i][:3], (1,2,0)))
        plt.title("Input Image"); plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(y_true[i][0], cmap='gray')
        plt.title("Ground Truth"); plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(y_pred[i][0]>0.5, cmap='gray')
        plt.title("Predicted"); plt.axis('off')
        plt.show()

# -----------------------------
# Training
# -----------------------------
def train(resume=False):
    set_seed()
    ensure_dir(CHECKPOINT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, in_ch = build_loaders()
    model = UNet(in_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = BCEDiceLoss()

    start_epoch = 1
    best_val_dice = 0
    patience = 0

    # --- Resume if requested ---
    if resume and os.path.exists(LAST_PATH):
        checkpoint_data = torch.load(LAST_PATH, map_location=device)
        if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
            # new format
            model.load_state_dict(checkpoint_data["model_state"])
            optimizer.load_state_dict(checkpoint_data["optimizer_state"])
            start_epoch = checkpoint_data.get("epoch", 1) + 1
            best_val_dice = checkpoint_data.get("best_val_dice", 0)
            patience = checkpoint_data.get("patience", 0)
            print(f"Resumed training from epoch {start_epoch}")
        else:
            # old format
            model.load_state_dict(checkpoint_data)
            print("Loaded old checkpoint, starting from epoch 1")

    # --- Training loop ---
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y) / ACCUM_STEPS
            loss.backward()
            if (i + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * ACCUM_STEPS

        train_loss = running_loss / len(train_loader)
        val_dice, val_iou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Val Dice={val_dice:.4f} Val IoU={val_iou:.4f}")

        # Save last checkpoint
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_dice": best_val_dice,
            "patience": patience
        }, LAST_PATH)

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_val_dice,
                "patience": patience
            }, BEST_PATH)
            patience = 0
        else:
            patience += 1

        if patience >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

    # --- Test evaluation ---
    test_dice, test_iou = evaluate(model, test_loader, device)
    print(f"Test Dice={test_dice:.4f} Test IoU={test_iou:.4f}")

    # --- Show some predictions ---
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    show_predictions(x, y, probs, n=3)
if __name__ == "__main__":
    train(resume=True)
