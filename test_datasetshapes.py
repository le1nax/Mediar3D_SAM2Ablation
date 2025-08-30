from monai.data import Dataset, DataLoader
from monai.transforms import LoadImaged, EnsureChannelFirstd
from pathlib import Path
from collections import Counter
import torch
import numpy as np
from PIL import Image

# adjust paths here
DATA_FOLDER = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/DSBowl2018/data-science-bowl-2018/stage1_train/CTC_format/01_tiff"

IMAGE_EXT = ".tiff"
LABEL_EXT = ".tiff"  # adjust if label files have a different extension

# find all images and labels
images = sorted(Path(DATA_FOLDER).rglob(f"*{IMAGE_EXT}"))
labels = sorted(Path(DATA_FOLDER).rglob(f"*{LABEL_EXT}"))

print(f"Found {len(images)} images")
print(f"Found {len(labels)} labels")

image_shapes = Counter()
label_shapes = Counter()

for img_path, lbl_path in zip(images, labels):
    img = torch.tensor(np.array(Image.open(img_path)))
    lbl = torch.tensor(np.array(Image.open(lbl_path)))

    # print some debug info
    print(f"Image: {img_path.name}, shape: {img.shape}")
    print(f"Label: {lbl_path.name}, shape: {lbl.shape}")

    image_shapes[img.shape] += 1
    label_shapes[lbl.shape] += 1

print("\n📊 Summary:")
print("Image spatial dims counts:", image_shapes)
print("Label spatial dims counts:", label_shapes)