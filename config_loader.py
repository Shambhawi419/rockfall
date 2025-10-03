# config_loader.py
import json
from pathlib import Path

# -----------------------------
# Load config.json
# -----------------------------
CONFIG_PATH = Path(__file__).parent / "config.json"

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

# -----------------------------
# Map config keys to training variables
# -----------------------------
# Base directory for dataset
DATA_DIR = str(Path(cfg["train_img_dir"]).parent.parent)  # parent of TrainData/img

# Dataset paths
TRAIN_IMG_DIR = cfg["train_img_dir"]
TRAIN_MASK_DIR = cfg["train_mask_dir"]
VALID_IMG_DIR = cfg["valid_img_dir"]
VALID_MASK_DIR = cfg["valid_mask_dir"]
TEST_IMG_DIR = cfg["test_img_dir"]
TEST_MASK_DIR = cfg["test_mask_dir"]

# DEM / topography
DEM_TIF = cfg.get("dem_tif", None)

# Checkpoints and logs
CHECKPOINT_DIR = cfg.get("checkpoints_dir", "./checkpoints")
BEST_PATH = cfg.get("best_checkpoint", str(Path(CHECKPOINT_DIR) / "best_multimodal.pt"))
LAST_PATH = cfg.get("last_checkpoint", str(Path(CHECKPOINT_DIR) / "last_multimodal.pt"))
LOGS_DIR = cfg.get("logs_dir", "./logs")

# Training hyperparameters
IMG_SIZE = cfg.get("img_size", 128)
BATCH_SIZE = cfg.get("batch_size", 2)
LR = cfg.get("learning_rate", 1e-3)
EPOCHS = cfg.get("epochs", 25)
ACCUM_STEPS = cfg.get("accum_steps", 1)
EARLY_STOP_PATIENCE = cfg.get("early_stop_patience", 5)
MAX_CHANNELS = cfg.get("max_channels", 12)
SEED = cfg.get("SEED", 42)
