import json
from pathlib import Path

# -----------------------------
# Load config.json
# -----------------------------
CONFIG_PATH = Path(__file__).parent / "config.json"

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

# -----------------------------
# Dataset paths
# -----------------------------
TRAIN_IMG_DIR = cfg["train_img_dir"]
TRAIN_MASK_DIR = cfg["train_mask_dir"]

VALID_IMG_DIR = cfg["valid_img_dir"]
VALID_MASK_DIR = cfg["valid_mask_dir"]

TEST_IMG_DIR = cfg["test_img_dir"]
TEST_MASK_DIR = cfg["test_mask_dir"]

DEM_TIF = cfg.get("dem_tif", None)

# -----------------------------
# Checkpoints and logs
# -----------------------------
CHECKPOINTS_DIR = cfg["checkpoints_dir"]
BEST_CHECKPOINT = cfg["best_checkpoint"]
LAST_CHECKPOINT = cfg["last_checkpoint"]
LOGS_DIR = cfg["logs_dir"]

# -----------------------------
# Training hyperparameters
# -----------------------------
BATCH_SIZE = cfg.get("batch_size", 2)
EPOCHS = cfg.get("epochs", 25)
IMG_SIZE = cfg.get("img_size", 128)
LR = cfg.get("learning_rate", 1e-3)
ACCUM_STEPS = cfg.get("accum_steps", 1)
EARLY_STOP_PATIENCE = cfg.get("early_stop_patience", 5)
MAX_CHANNELS = cfg.get("max_channels", 12)
SEED = cfg.get("seed", 42)
