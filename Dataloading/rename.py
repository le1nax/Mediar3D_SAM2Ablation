import os
import glob

# Set your directory path
folder_path = "/work/scratch/geiger/Datasets/CTC/sim3d/02_test"

# Get all files in the folder (adjust the pattern if necessary)
files = sorted(glob.glob(os.path.join(folder_path, "*")), key=lambda x: os.path.basename(x))

# Rename files sequentially
for idx, file in enumerate(files, start=1):
    new_filename = f"cell_{idx:05d}.tif"  # Format with 5-digit zero padding
    new_filepath = os.path.join(folder_path, new_filename)
    
    # Rename file
    os.rename(file, new_filepath)

# Set your directory path
folder_path = "/work/scratch/geiger/Datasets/CTC/sim3d/02_test_GT"

# Get all files in the folder (adjust the pattern if necessary)
files = sorted(glob.glob(os.path.join(folder_path, "*")), key=lambda x: os.path.basename(x))

# Rename files sequentially
for idx, file in enumerate(files, start=1):
    new_filename = f"cell_{idx:05d}_label.tiff"  # Format with 5-digit zero padding
    new_filepath = os.path.join(folder_path, new_filename)
    
    # Rename file
    os.rename(file, new_filepath)

print("Renaming completed!")
