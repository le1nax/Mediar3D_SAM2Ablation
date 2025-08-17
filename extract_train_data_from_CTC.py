import re
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from tifffile import TiffFile

# --- Visualization function ---
def show_QC_results_overlay(src_image, gt_image):
    """
    Show an overlay of the source image and ground truth mask.
    """
    norm = mcolors.Normalize(vmin=np.percentile(src_image, 1),
                             vmax=np.percentile(src_image, 99))
    mask_norm = mcolors.Normalize(vmin=0, vmax=1)

    plt.figure(figsize=(8, 8))
    plt.imshow(src_image, cmap='gray', norm=norm, interpolation='nearest')
    plt.imshow(gt_image, cmap='Greens', norm=mask_norm, alpha=0.5)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# --- Configuration ---
source_image_dir = Path("/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2020/Fluo-N3DL-TRIF/01")  # txxx.tif
source_label_dir = Path("/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2020/Fluo-N3DL-TRIF/01_GT/SEG")  # man_seg_xxx_zzz.tif
output_image_dir = Path("/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-TRIF/01_img_train")
output_label_dir = Path("/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-TRIF/01_label_train")

output_image_dir.mkdir(parents=True, exist_ok=True)
output_label_dir.mkdir(parents=True, exist_ok=True)

# Regex patterns
image_pattern = re.compile(r"^t(\d{3})\.tif$")
label_pattern = re.compile(r"^man_seg_(\d{3})_(\d{3})\.tif$")

# Step 1: list image volumes
image_files = {m.group(1): f for f in source_image_dir.iterdir()
               if (m := image_pattern.match(f.name))}
print(f"Found {len(image_files)} image volumes.")

# Step 2: group labels by timestamp
labels_by_timestamp = {}
for f in source_label_dir.iterdir():
    m = label_pattern.match(f.name)
    if m:
        ts, zslice = m.groups()
        labels_by_timestamp.setdefault(ts, []).append((int(zslice), f))
print(f"Found labels for {len(labels_by_timestamp)} timestamps.")

# Step 3: iterate and extract
for ts, label_info in labels_by_timestamp.items():
    if ts not in image_files:
        print(f"⚠ No source image found for timestamp {ts}, skipping.")
        continue

    img_path = image_files[ts]
    print(f"Processing timestamp {ts}...")

    with TiffFile(img_path) as tif:
        for z_idx, label_path in label_info:
            if z_idx < 0 or z_idx >= len(tif.pages):
                print(f"⚠ Z slice {z_idx} out of range for timestamp {ts}, skipping.")
                continue

            # Read only the required slice
            slice_img = tif.pages[z_idx].asarray()
            label_img = tifffile.imread(label_path)

            # Save
            img_out_name = f"cell_{ts}{z_idx:03d}.tif"
            lbl_out_name = f"cell_{ts}{z_idx:03d}_label.tif"
            tifffile.imwrite(output_image_dir / img_out_name, slice_img)
            tifffile.imwrite(output_label_dir / lbl_out_name, label_img)

            # QC overlay
            #show_QC_results_overlay(slice_img, label_img)

print("Done.")
