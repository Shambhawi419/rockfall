# build_multimodal_h5_cache_landslide4_fixed.py
import os
import re
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# -----------------------------
# Config (edit if needed)
# -----------------------------
IMAGE_DIR = r"D:\rockfall_ai\data\drone_img"      # base dir for images (CSV has relative paths)
MASK_DIR  = r"D:\rockfall_ai\data\drone_mask"     # base dir for masks
CSV_DIR   = r"D:\rockfall_ai\data\clean_dataset"  # train.csv / val.csv / test.csv
OUTPUT_DIR = r"D:\rockfall_ai\data\multimodal_h5" # output multimodal .h5 files
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEATHER_DIM = 4
GEOTECH_DIM = 3
DTYPE = np.float32

# -----------------------------
# Helpers
# -----------------------------
id_re = re.compile(r'(\d+)')
def find_numeric_id(fname):
    m = id_re.search(fname)
    return m.group(1) if m else None

def read_first_dataset_from_h5(path):
    with h5py.File(path, "r") as f:
        for key in ("drone", "img", "image", "data"):
            if key in f:
                return f[key][()], key
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No datasets in {path}")
        key = keys[0]
        return f[key][()], key

def normalize_image_array(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] <= 4 and arr.shape[1] > 4 and arr.shape[2] > 4:
            arr = np.transpose(arr, (1,2,0))
    elif arr.ndim == 2:
        arr = arr[..., None]
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.1:
            arr = arr / 255.0
    return arr

def read_mask_array(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
        arr = np.transpose(arr, (1,2,0))
    arr = arr.astype(np.float32)
    if arr.max() > 1.1:
        arr = arr / 255.0
    return arr

def generate_fake_dem_and_slope(H, W, seed=None):
    rng = np.random.default_rng(seed)
    dem = rng.uniform(0.0, 3000.0, size=(H, W)).astype(DTYPE)
    slope = rng.uniform(0.0, 60.0, size=(H, W)).astype(DTYPE)
    return dem, slope

def generate_fake_weather(seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(0.5, 0.2, size=(WEATHER_DIM,)).astype(DTYPE)

def generate_fake_geotech(seed=None):
    rng = np.random.default_rng(seed)
    return rng.random(GEOTECH_DIM).astype(DTYPE)

def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

# -----------------------------
# Main builder
# -----------------------------
def find_file_recursively(base_dir, filename):
    """Search recursively for a file and return full path."""
    matches = list(Path(base_dir).rglob(filename))
    return str(matches[0]) if matches else None

def build_for_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['mask'] = df.get('mask', '').fillna('')

    created_list = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(csv_path)}"):
        filename = str(row["filename"]).strip()
        maskname  = str(row["mask"]).strip()

        img_path = find_file_recursively(IMAGE_DIR, filename)
        if not img_path:
            print(f"⚠️ Skipping (no image): {filename}")
            skipped += 1
            continue

        # Mask lookup
        mask_path = find_file_recursively(MASK_DIR, maskname) if maskname else None
        if not mask_path:
            idnum = find_numeric_id(filename)
            if idnum:
                mask_path = find_file_recursively(MASK_DIR, f"mask_{idnum}.h5")

        # Read image
        try:
            img_arr, img_key = read_first_dataset_from_h5(img_path)
            drone = normalize_image_array(img_arr)
        except Exception as e:
            print(f"⚠️ Error reading image H5 {img_path} -> {e}")
            skipped += 1
            continue

        H, W = drone.shape[:2]

        # Read mask or create zero mask
        if mask_path:
            try:
                mask_arr, mask_key = read_first_dataset_from_h5(mask_path)
                mask = read_mask_array(mask_arr)
                if mask.ndim == 3 and mask.shape[-1] == 1:
                    mask = mask[..., 0]
                if mask.shape[:2] != (H, W):
                    mask = np.zeros((H, W), dtype=DTYPE)
            except:
                mask = np.zeros((H, W), dtype=DTYPE)
        else:
            mask = np.zeros((H, W), dtype=DTYPE)

        # Generate fake DEM, slope, weather, geotech
        seed = idx
        dem, slope = generate_fake_dem_and_slope(H, W, seed=seed)
        weather = generate_fake_weather(seed=seed)
        geotech = generate_fake_geotech(seed=seed)

        # Save multimodal H5
        out_name = filename.replace(".h5", "_multi.h5")
        out_path = os.path.join(OUTPUT_DIR, out_name)
        ensure_parent_dir(out_path)

        try:
            with h5py.File(out_path, "w") as f:
                f.create_dataset("drone", data=drone, compression="gzip")
                f.create_dataset("mask", data=mask, compression="gzip")
                f.create_dataset("dem", data=dem, compression="gzip")
                f.create_dataset("slope", data=slope, compression="gzip")
                f.create_dataset("weather", data=weather, compression="gzip")
                f.create_dataset("geotech", data=geotech, compression="gzip")
            created_list.append(out_path)
        except Exception as e:
            print(f"⚠️ Error saving {out_path} -> {e}")
            skipped += 1

    print(f"Done for {os.path.basename(csv_path)} — created: {len(created_list)}, skipped: {skipped}")
    return created_list

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    total_created = 0
    for split in ["train.csv", "val.csv", "test.csv"]:
        csv_path = os.path.join(CSV_DIR, split)
        if os.path.exists(csv_path):
            created = build_for_csv(csv_path)
            total_created += len(created)
        else:
            print(f"⚠️ CSV missing: {csv_path}")

    print(f"\nTotal multimodal files created: {total_created}")
    print(f"Saved into: {os.path.abspath(OUTPUT_DIR)}")












