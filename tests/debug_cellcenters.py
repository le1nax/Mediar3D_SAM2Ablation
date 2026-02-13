import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import dilation, disk
from scipy.ndimage import binary_dilation



# Set paths
cellcenter_dir = Path("/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/fs_train_cellcenters")
mask_dir = Path("/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/fs_train_masks")


# Load all cell center masks
cellcenter_files = sorted(cellcenter_dir.glob("cell_centers_*.tiff"))

for cc_file in cellcenter_files:
    # Extract the numeric ID
    stem_id = cc_file.stem.split("_")[-1]  # e.g., '00015'

    # Build corresponding full mask filename
    mask_file = mask_dir / f"cell_{stem_id}_label.tiff"

    if not mask_file.exists():
        print(f"Mask not found for: {cc_file.name} -> expected: {mask_file.name}")
        continue

    # Load images
    cellcenter = imread(cc_file)
    mask = imread(mask_file)

    # Ensure grayscale
    if cellcenter.ndim > 2:
        cellcenter = cellcenter[..., 0]
    if mask.ndim > 2:
        mask = mask[..., 0]

    # === Debug print for cell center mask ===
    nonzero_count = np.count_nonzero(cellcenter)
    print(f"{cc_file.name} → Nonzero pixels in cellcenter: {nonzero_count}")

    # === Dilate cell centers for better visibility ===
    dilated_centers = binary_dilation(cellcenter > 0, iterations=2)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(mask, cmap="gray")
    axs[0].set_title("Full Mask")

    axs[1].imshow(mask, cmap="gray")
    axs[1].imshow(dilated_centers, cmap="Reds", alpha=0.6)
    axs[1].set_title("Dilated Cell Centers Overlay")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.io import imread
# from scipy.ndimage import binary_dilation
# from pathlib import Path

# # === Paths to your 3D .tiff files ===
# cellcenter_path = Path("/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/fs_train_cellcenters")
# mask_path = Path("/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/fs_train_masks")

# # === Load 3D images ===
# cellcenter_3d = imread(cellcenter_path)
# mask_3d = imread(mask_path)

# # === Check for shape compatibility ===
# assert cellcenter_3d.shape == mask_3d.shape, "Shape mismatch between center and mask volumes!"

# # === Visualize slices ===
# for i in range(cellcenter_3d.shape[0]):
#     center_slice = cellcenter_3d[i]
#     mask_slice = mask_3d[i]

#     # === Debug: count nonzero pixels in current slice ===
#     nonzero_count = np.count_nonzero(center_slice)
#     print(f"Slice {i:03d} — Cell Center Nonzero Count: {nonzero_count}")

#     # === Dilate the center for visibility ===
#     dilated_center = binary_dilation(center_slice > 0, iterations=3)

#     # === Plot ===
#     fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    
#     axs[0].imshow(mask_slice, cmap="gray")
#     axs[0].set_title(f"Mask Slice {i}")
    
#     axs[1].imshow(mask_slice, cmap="gray")
#     axs[1].imshow(dilated_center, cmap="Reds", alpha=0.5)
#     axs[1].set_title(f"Overlay Dilated Centers Slice {i}")
    
#     for ax in axs:
#         ax.axis("off")
    
#     plt.tight_layout()
#     plt.show()