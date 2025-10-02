import os
import h5py
import numpy as np

# Paths to your dataset
img_dir = r"D:\rockfall_ai\data\drone_img\landslide4sense\TrainData\img"
mask_dir = r"D:\rockfall_ai\data\drone_img\landslide4sense\TrainData\mask"

# Function to load a single HDF5 file
def load_h5_file(file_path, dataset_name=None):
    """
    Load data from an HDF5 file.
    If dataset_name is None, it loads the first dataset in the file.
    """
    with h5py.File(file_path, 'r') as f:
        if dataset_name is None:
            dataset_name = list(f.keys())[0]
        data = f[dataset_name][:]
    return data

# Get sorted list of files to maintain alignment
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.h5')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.h5')])

# Ensure the number of images and masks match
assert len(img_files) == len(mask_files), "Number of images and masks do not match!"

# Load all images and masks into NumPy arrays
images = []
masks = []

for img_file, mask_file in zip(img_files, mask_files):
    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)
    
    img_data = load_h5_file(img_path)
    mask_data = load_h5_file(mask_path)
    
    # Expand dims if grayscale (H, W) -> (H, W, 1)
    if img_data.ndim == 2:
        img_data = np.expand_dims(img_data, axis=-1)
    if mask_data.ndim == 2:
        mask_data = np.expand_dims(mask_data, axis=-1)
    
    images.append(img_data)
    masks.append(mask_data)

# Convert lists to NumPy arrays
images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

print(f"Loaded {images.shape[0]} images of shape {images.shape[1:]}")
print(f"Loaded {masks.shape[0]} masks of shape {masks.shape[1:]}")









