# -------------------------------
# 0️⃣ Imports
# -------------------------------
import os
import ee
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import rasterio

# -------------------------------
# 1️⃣ Authenticate GEE
# -------------------------------
ee.Authenticate()   # <-- Run this once. It opens browser login.
ee.Initialize()

# -------------------------------
# 2️⃣ Set Paths
# -------------------------------
BASE_DIR = "data"
IMG_DIR = r"D:\Datasets\Landslide4Sense\img"
MASK_DIR = r"D:\Datasets\Landslide4Sense\mask"
os.makedirs(BASE_DIR, exist_ok=True)

# -------------------------------
# 3️⃣ Fetch DEM + Slope from GEE
# -------------------------------
aoi = ee.Geometry.Rectangle([85.0, 27.5, 86.0, 28.5])  # change AOI

dem = ee.Image('USGS/SRTMGL1_003').clip(aoi)
slope = ee.Terrain.slope(dem)

def export_gee_image(img, filename):
    path = os.path.join(BASE_DIR, f"{filename}.tif")
    url = img.getDownloadURL({
        'scale': 30,
        'crs': 'EPSG:4326',
        'region': aoi.getInfo()
    })
    import requests
    r = requests.get(url)
    with open(path, 'wb') as f:
        f.write(r.content)
    print(f"✅ Saved {filename}.tif")
    return path

dem_path = export_gee_image(dem, 'dem')
slope_path = export_gee_image(slope, 'slope')

# -------------------------------
# 4️⃣ Generate Synthetic Weather
# -------------------------------
def generate_synthetic_weather(rows=500):
    np.random.seed(42)
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="H")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "rainfall_mm": np.random.gamma(2,1,rows),
        "temperature_C": 15 + 10*np.random.randn(rows),
        "seismic_activity": np.random.rand(rows),
        "soil_moisture": np.clip(0.2 + 0.1*np.random.randn(rows), 0,1)
    })
    df.to_csv(os.path.join(BASE_DIR,'synthetic_weather.csv'), index=False)
    print("✅ Saved synthetic_weather.csv")
    return df

weather_df = generate_synthetic_weather()

# -------------------------------
# 5️⃣ Load DEM & Slope as Torch Tensors
# -------------------------------
def load_dem_slope(dem_path, slope_path, img_size=224):
    def read_tif(path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        t = torch.from_numpy(arr).unsqueeze(0)
        t = F.interpolate(t.unsqueeze(0), size=(img_size,img_size), mode='bilinear').squeeze(0)
        return t
    dem_tensor = read_tif(dem_path)
    slope_tensor = read_tif(slope_path)
    return dem_tensor, slope_tensor

dem_tensor, slope_tensor = load_dem_slope(dem_path, slope_path)

# -------------------------------
# 6️⃣ Create Multimodal Dataset
# -------------------------------
class MultimodalSegDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, dem_tensor, slope_tensor, weather_df, img_size=224):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dem = dem_tensor
        self.slope = slope_tensor
        self.weather = weather_df
        self.img_size = img_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load drone image
        with h5py.File(os.path.join(self.img_dir, row['filename']), 'r') as f:
            img = f['img'][:,:,:3].astype(np.float32)/255.0
        img = torch.from_numpy(img).permute(2,0,1)
        # Load mask
        with h5py.File(os.path.join(self.mask_dir, row['mask']), 'r') as f:
            mask = torch.from_numpy(f['mask'][:]).unsqueeze(0).float()
        # DEM & slope resized
        dem_resized = F.interpolate(self.dem.unsqueeze(0), size=(self.img_size,self.img_size)).squeeze(0)
        slope_resized = F.interpolate(self.slope.unsqueeze(0), size=(self.img_size,self.img_size)).squeeze(0)
        # Weather tensor
        w_row = self.weather.iloc[idx % len(self.weather)]
        weather_tensor = torch.tensor([w_row['rainfall_mm'], w_row['temperature_C'], 
                                       w_row['seismic_activity'], w_row['soil_moisture']], dtype=torch.float32)
        weather_tensor = weather_tensor.view(-1,1,1).repeat(1,self.img_size,self.img_size)
        # Combine channels: RGB(3)+DEM(1)+Slope(1)+Weather(4)=9
        x = torch.cat([img, dem_resized.unsqueeze(0), slope_resized.unsqueeze(0), weather_tensor], dim=0)
        return x, mask

# -------------------------------
# 7️⃣ Example Usage
# -------------------------------
CSV_FILE = os.path.join(BASE_DIR,'drone_labels.csv')  # make sure you have this CSV
dataset = MultimodalSegDataset(CSV_FILE, IMG_DIR, MASK_DIR, dem_tensor, slope_tensor, weather_df)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for x,y in loader:
    print("Input shape:", x.shape)   # [B,9,H,W]
    print("Mask shape:", y.shape)    # [B,1,H,W]
    break


