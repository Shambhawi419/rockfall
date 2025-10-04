
# Rockfall Multimodal UNet for Landslide Segmentation

## Description
This project implements a **Laptop-friendly Multimodal UNet** for landslide segmentation using drone imagery.  
It integrates multiple modalities:
- **Drone RGB images**
- **Digital Elevation Models (DEM)**
- **Synthetic sensor and weather data**  

The model handles **empty masks**, supports **train/validation/test splits**, and provides evaluation metrics and visualization for predictions.

---

## Project Structure
```

rockfall_ai/
├─ train_fusion_full.py       # Main training script
├─ config.json                # Configuration for paths and hyperparameters
├─ README.md                  # Project documentation
├─ requirements.txt           # Python dependencies
├─ checkpoints/               # Folder for saving model checkpoints
├─ data/
│   ├─ TrainData/
│   │   ├─ img/               # Training images in .h5 format
│   │   └─ mask/              # Corresponding masks in .h5 format
│   ├─ ValidData/
│   │   ├─ img/               # Validation images
│   │   └─ mask/              # Validation masks
│   └─ TestData/
│       ├─ img/               # Test images
│       └─ mask/              # Test masks

````

---

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
````

Key libraries include:

* `torch` / `torchvision` – PyTorch for deep learning
* `numpy` – numerical computations
* `h5py` – reading H5 dataset files
* `matplotlib` – plotting and visualization
* `scikit-learn` – train/test splitting and metrics
* `rasterio` – optional, for DEM processing
* (Optional) GPU for faster training

---

## Configuration (`config.json`)

All paths and hyperparameters are managed via `config.json`. Example:

```json
{
  "train_img_dir": "data/TrainData/img",
  "train_mask_dir": "data/TrainData/mask",
  "valid_img_dir": "data/ValidData/img",
  "valid_mask_dir": "data/ValidData/mask",
  "test_img_dir": "data/TestData/img",
  "test_mask_dir": "data/TestData/mask",
  "dem_tif": "data/clean_dataset/dem/mine_dem.tif",
  "checkpoints_dir": "checkpoints",
  "best_checkpoint": "checkpoints/best_multimodal.pt",
  "last_checkpoint": "checkpoints/last_multimodal.pt",
  "logs_dir": "logs",
  "batch_size": 2,
  "epochs": 25,
  "img_size": 128,
  "learning_rate": 0.001,
  "accum_steps": 2,
  "early_stop_patience": 5,
  "max_channels": 12,
  "seed": 42
}
```

---

## Usage

### Training

```bash
python train_fusion_full.py
```

* Automatically **resumes training** if a checkpoint exists.
* Supports **gradient accumulation** and **early stopping**.

### Evaluation

* Computes **Dice coefficient** and **IoU** on validation and test sets.
* Visualizes predicted vs. ground truth masks.

---

## Features

* **Multimodal input:** RGB + DEM + weather/sensor channels
* **Empty mask handling:** Prevents training on empty masks
* **Config-driven:** All paths and hyperparameters in `config.json`
* **Checkpointing:** Saves best and last model
* **Visualization:** Compare predictions with ground truth

---

## Dataset

⚠️ Note: Datasets are not included in this repo. Please download the Landslide4 dataset from [Zenodo](https://zenodo.org/records/10463239) and place it in the `data/` folder.


* Images and masks are stored in **H5 format**.
* Masks are binary and resized to **128x128**.

```
  

