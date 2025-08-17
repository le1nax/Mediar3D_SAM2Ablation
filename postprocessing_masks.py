import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_dilation
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))


def filter_false_positives_intermediate(pred_mask, cellcenters):
    """
    Keep only those cell instances in pred_mask that contain at least one non-zero pixel in cellcenters.

    Shows intermediate info and plots for each instance.

    Returns:
        ndarray: Filtered instance mask with only valid cell instances.
    """
    assert pred_mask.shape == cellcenters.shape, "Shape mismatch between pred_mask and cellcenters"

    filtered_mask = np.zeros_like(pred_mask, dtype=pred_mask.dtype)
    instance_ids = np.unique(pred_mask)
    instance_ids = instance_ids[instance_ids != 0]

    for inst_id in instance_ids:
        inst_mask = (pred_mask == inst_id)
        has_center = np.any(cellcenters[inst_mask])

        #print(f"Instance {inst_id}: center pixels present? {has_center}")
        if has_center:
            # Find slices where this instance exists
            #slices_with_inst = np.where(inst_mask.any(axis=(1, 2)))[0]
            # if len(slices_with_inst) > 0:
               # z = slices_with_inst[0]

                # # Extract the instance mask for this slice
                # inst_slice_mask = (pred_mask[z] == inst_id)

                # # Extract corresponding cellcenter slice and mask by instance mask
                # cellcenter_slice = cellcenters[z]

                # Dilate the cell centers inside this instance mask for visibility
                #center_mask_instance = inst_slice_mask & (cellcenter_slice > 0)
                #dilated_centers = binary_dilation(center_mask_instance, iterations=3)

                # # Plotting
                # plt.figure(figsize=(8, 4))

                # plt.subplot(1, 2, 1)
                # plt.imshow(inst_slice_mask, cmap='nipy_spectral')
                # plt.title(f"Instance {inst_id} Mask Slice {z}")
                # plt.axis('off')

                # plt.subplot(1, 2, 2)
                # #plt.imshow(inst_slice_mask, cmap='gray')  # Show instance as background
                # plt.imshow(dilated_centers, cmap='Reds')  # Overlay dilated centers in red
                # plt.title(f"Dilated Cell Centers (Instance {inst_id})")
                # plt.axis('off')

                # plt.suptitle(f"Instance {inst_id} — Kept: {has_center}")
                # plt.show()

            filtered_mask[inst_mask] = inst_id
    return filtered_mask


# === Load files ===
mask_path = Path("../../Datasets/CTC/test_images/zc2dg/fs_inference_res_100e_Mediarclean/cell_0000_label.tiff")
center_path = Path("../../Datasets/CTC/test_images/zc2dg/fs_inference_cellcenters/cell_0000_cellcenter.tiff")

masks = io.imread(mask_path)
cellcenters = io.imread(center_path)

# === Filter with intermediate visualization ===
filtered = filter_false_positives_intermediate(masks, cellcenters)

# === Save result ===
save_path = Path("../../Datasets/CTC/test_images/zc2dg/fs_inference_res_100e_Mediarclean/cell_0000_label.tiff")
save_path.parent.mkdir(parents=True, exist_ok=True)
io.imsave(save_path, filtered.astype(np.uint16))