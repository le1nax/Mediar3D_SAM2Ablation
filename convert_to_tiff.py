import os
from pathlib import Path
from PIL import Image
import numpy as np

def convert_png_to_tiff_flat(input_folder, output_folder=None, recursive=False):
    """
    Convert PNG images to single-channel TIFF images by removing the channel dimension.
    
    Args:
        input_folder (str or Path): Folder containing PNG images.
        output_folder (str or Path, optional): Where to save TIFFs. If None, saves in the same folder.
        recursive (bool): If True, process subfolders as well.
    """
    input_folder = Path(input_folder)
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    pattern = "**/*.png" if recursive else "*.png"
    png_files = list(input_folder.glob(pattern))
    
    if not png_files:
        print("No PNG files found in", input_folder)
        return
    
    for png_path in png_files:
        if output_folder:
            rel_path = png_path.relative_to(input_folder)
            tiff_path = output_folder / rel_path.with_suffix(".tiff")
            tiff_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            tiff_path = png_path.with_suffix(".tiff")
        
        # Open PNG and convert to numpy array
        img = Image.open(png_path)
        arr = np.array(img)
        
        # If image has channels, reduce to single channel (take first channel)
        if arr.ndim == 3:
            arr = arr[..., 0]  # take the first channel
        
        # Save as TIFF
        img_tiff = Image.fromarray(arr)
        img_tiff.save(tiff_path, format="TIFF")
        print(f"{png_path} → {tiff_path}")

if __name__ == "__main__":
    input_dir = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/DSBowl2018/data-science-bowl-2018/stage1_train/CTC_format/01"  # change this
    output_dir = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/DSBowl2018/data-science-bowl-2018/stage1_train/CTC_format/01_tiff"  # or None to overwrite in place
    convert_png_to_tiff_flat(input_dir, output_dir, recursive=True)