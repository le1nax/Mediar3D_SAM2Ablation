import numpy as np
import pandas as pd
import tifffile as tif
import argparse
import os
from collections import OrderedDict
from tqdm import tqdm

from train_tools.utils import ConfLoader, pprint_config
from train_tools.measures import evaluate_f1_score_cellseg, evaluate_metrics_cellseg

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from matplotlib.widgets import TextBox
from skimage import io
import pandas as pd

def z_project_cellcenters(cellcenters):
    """
    Projects all cell centers along the Z-axis so that every Z-slice contains all cell centers.

    Args:
        cellcenters (ndarray): 3D binary mask of shape (Z, H, W)

    Returns:
        ndarray: 3D binary mask with projected centers in every slice
    """
    # Project along Z-axis: (H, W)
    projected = np.any(cellcenters > 0, axis=0).astype(np.uint8)
    
    # Tile across Z-axis: (Z, H, W)
    projected_volume = np.repeat(projected[None, :, :], cellcenters.shape[0], axis=0)

    return projected_volume

def project_lower_to_upper(cellcenters, slice_index, axis=0):
    """
    Projects all cell centers from the lower part (below slice_index along given axis)
    to the first slice of the upper part, to compensate for center loss due to cropping.

    Args:
        cellcenters (ndarray): 3D binary mask of shape (Z, H, W)
        slice_index (int): Index at which the volume is split into lower and upper parts.
        axis (int): Axis along which to perform the split (0=Z, 1=Y, 2=X)

    Returns:
        ndarray: Modified 3D binary mask for the upper half with projected centers.
    """
    if axis not in [0, 1, 2]:
        raise ValueError("Axis must be 0 (Z), 1 (Y), or 2 (X)")

    # Move axis to front for uniform processing
    data = np.moveaxis(cellcenters, axis, 0)  # shape -> (D, other1, other2)

    # Split and copy upper part
    upper = np.copy(data[slice_index:])  # shape (D_upper, other1, other2)

    # Projection from lower part
    lower_projection = np.any(data[:slice_index] > 0, axis=0).astype(np.uint8)  # shape (other1, other2)

    # Project onto the first slice of upper part
    if upper.shape[0] > 0:
        upper[0] = np.logical_or(upper[0], lower_projection).astype(np.uint8)

    # Reconstruct full output in original axis order
    result = np.zeros_like(data, dtype=np.uint8)
    result[slice_index:] = upper
    result = np.moveaxis(result, 0, axis)

    return result

def crop_along_axis(volume, slice_obj, axis=0):
    """
    Crops a 3D volume along a specified axis using a provided slice object.

    Args:
        volume (ndarray): 3D array (e.g., shape (Z, H, W))
        slice_obj (slice): A Python slice object (e.g., slice(0, 64), slice(None, 32))
        axis (int): Axis along which to apply the slice (0=Z, 1=Y, 2=X)

    Returns:
        ndarray: Cropped 3D volume
    """
    assert volume.ndim == 3, "Input must be a 3D array"
    assert 0 <= axis <= 2, "Axis must be 0 (Z), 1 (Y), or 2 (X)"
    assert isinstance(slice_obj, slice), "slice_obj must be a slice object"

    # Create full slice for each axis
    slicer = [slice(None)] * 3
    slicer[axis] = slice_obj

    return volume[tuple(slicer)]

def main(args):

    slice_index = args.eval_setups.slice_index
    axis = args.eval_setups.axis

    # Output directory to save slices
    output_img_dir = args.eval_setups.fs_train_img
    output_val_dir = args.eval_setups.fs_train_masks
    #output_cellcenter_dir = args.eval_setups.fs_train_cellcenters

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    #os.makedirs(output_cellcenter_dir, exist_ok=True)

    # Output directory to save inference image
    output_inference_img_dir = args.eval_setups.fs_inference_img
    output_inference_val_dir = args.eval_setups.fs_inference_masks
    #output_inference_cellcenters_dir = args.eval_setups.fs_inference_cellcenters

    os.makedirs(output_inference_img_dir, exist_ok=True)
    os.makedirs(output_inference_val_dir, exist_ok=True)
    #os.makedirs(output_inference_cellcenters_dir, exist_ok=True)


    
    # Get files from the paths
    gt_path = args.eval_setups.gt_path
    #pred_path = args.eval_setups.pred_path
    img_path = args.eval_setups.img_path
    #cellcenters_path = args.eval_setups.cellcenter_path


    img_names = sorted(os.listdir(img_path))
    gt_names = sorted(os.listdir(gt_path))
    #cellcenter_names = sorted(os.listdir(cellcenters_path))


    buffer = 10  # pixels to extend beyond bounding box

    for i in tqdm(range(len(img_names))):
        # Load images
        gt = tif.imread(os.path.join(gt_path, gt_names[i]))
        img = tif.imread(os.path.join(img_path, img_names[i]))
        #cellcenters = tif.imread(os.path.join(cellcenters_path, cellcenter_names[i]))

        assert gt.shape == img.shape, f"Shape mismatch: {img.shape} vs {gt.shape}"

        # Find bounding box of the label
        nonzero = np.argwhere(gt > 0)
        if nonzero.size == 0:
            print(f"Warning: No label found in {gt_names[i]}")
            continue

        zmin, ymin, xmin = nonzero.min(axis=0)
        zmax, ymax, xmax = nonzero.max(axis=0)

        # Add buffer and clip to image bounds
        zmin = max(zmin - buffer, 0)
        ymin = max(ymin - buffer, 0)
        xmin = max(xmin - buffer, 0)

        zmax = min(zmax + buffer + 1, gt.shape[0])  # +1 to include the max index
        ymax = min(ymax + buffer + 1, gt.shape[1])
        xmax = min(xmax + buffer + 1, gt.shape[2])

        # Crop image and label
        img_crop = img[zmin:zmax, ymin:ymax, xmin:xmax]
        gt_crop = gt[zmin:zmax, ymin:ymax, xmin:xmax]

        #img_crop = crop_along_axis(img_crop, slice(0,slice_index), axis=2)
        #gt_crop = crop_along_axis(gt_crop, slice(0,slice_index), axis=2)

        save_3d_slices_in_all_directions(img_crop, gt_crop, output_img_dir, output_val_dir)


        # # Save with consistent zero-padded filename
        # filename = f"cell_{i:05d}.tiff"
        # filepath = os.path.join(output_inference_img_dir, filename)
        # tif.imwrite(filepath, img_crop)
        # print(f"Saved '{filepath}'.")

        # filename = f"cell_{i:05d}_label.tiff"
        # filepath = os.path.join(output_inference_val_dir, filename)
        # tif.imwrite(filepath, gt_crop)
        # print(f"Saved '{filepath}'.")


        # ###crop 3d few shot inference image

        # img_infer = img[slice_index:]
        # gt_infer = gt[slice_index:]
       
        # #upper_cellcenters_proj = project_lower_to_upper(cellcenters, slice_index, axis=axis)

        # infer_img_path = os.path.join(output_inference_img_dir, f"cell_{i:04d}.tiff")
        # infer_gt_path = os.path.join(output_inference_val_dir, f"cell_{i:04d}_label.tiff")
        # #infer_cellcenter_path = os.path.join(output_inference_cellcenters_dir, f"infer_{i:03d}_cellcenter.tiff")
        

        # tif.imwrite(infer_img_path, img_infer)
        # tif.imwrite(infer_gt_path, gt_infer)
        # #tif.imwrite(infer_cellcenter_path, upper_cellcenters_proj)


    # print(f"Saved inference volume to '{infer_img_path}' and '{infer_gt_path}'.")

def save_3d_slices_in_all_directions(img_crop, gt_crop, output_img_dir, output_val_dir, starting_index=0):
    """
    Slices 3D images and masks in Z, Y, and X directions, saves 2D slices with unique filenames,
    and skips slices where the corresponding mask is completely empty.

    Args:
        img_crop (ndarray): 3D image, shape (Z, H, W)
        gt_crop (ndarray): 3D mask, shape (Z, H, W)
        output_img_dir (str): Path to save 2D image slices
        output_val_dir (str): Path to save 2D mask slices
        starting_index (int): Index to start filename numbering from
    """
    assert img_crop.shape == gt_crop.shape, "Image and mask must have the same shape"
    idx = starting_index
    skipped = 0

    directions = ['Z', 'Y', 'X']
    slices = [
        img_crop,                            # Z: axial
        np.transpose(img_crop, (1, 0, 2)),   # Y: coronal (H, Z, W)
        np.transpose(img_crop, (2, 0, 1)),   # X: sagittal (W, Z, H)
    ]
    masks = [
        gt_crop,
        np.transpose(gt_crop, (1, 0, 2)),
        np.transpose(gt_crop, (2, 0, 1)),
    ]

    for d, (img_slices, gt_slices) in enumerate(zip(slices, masks)):
        for i in range(img_slices.shape[0]):
            img_slice_2d = img_slices[i]
            gt_slice_2d = gt_slices[i]

            if np.all(gt_slice_2d == 0):
                skipped += 1
                continue

            img_filename = f"cell_{idx:05d}.tiff"
            gt_filename = f"cell_{idx:05d}_label.tiff"

            tif.imwrite(os.path.join(output_img_dir, img_filename), img_slice_2d)
            tif.imwrite(os.path.join(output_val_dir, gt_filename), gt_slice_2d)

            print(f"[{directions[d]}] Saved slice {idx:05d}")
            idx += 1

    total_saved = idx - starting_index
    print(f"\nTotal saved slices: {total_saved}")
    print(f"Total skipped slices (empty masks): {skipped}")


def show_QC_results(img_path, pred_path, gt_path, cellseg_metric, slice_index=25):
        print("now comes the plot")

        source_files = [f for f in os.listdir(img_path) if f.endswith('.tiff') or f.endswith('.tif')]
        prediction_files = [f for f in os.listdir(pred_path) if f.endswith('.tiff') or f.endswith('.tif')]
        target_files = [f for f in os.listdir(gt_path) if f.endswith('.tiff') or f.endswith('.tif')]

        if len(source_files) != len(target_files) or len(source_files) != len(prediction_files):
            raise ValueError("The number of source and target files does not match.")
       
        # Initialize arrays to hold the images
        images_list = []

        # Load all the images from the source and target directories
        for src_file, pred_file, gt_file in zip(source_files, prediction_files, target_files):
            src_image = io.imread(os.path.join(img_path, src_file))
            pred_image = io.imread(os.path.join(pred_path, pred_file))
            gt_image = io.imread(os.path.join(gt_path, gt_file))
            images_list.append((src_image, pred_image, gt_image))


        # Convert list to a 4D numpy array (images, z, predicted_images, source_images)
        source_images = np.array([item[0] for item in images_list])  # Source images
        predicted_images = np.array([item[1] for item in images_list])  # Target images
        ground_truth_images = np.array([item[2] for item in images_list])  # GT images

        # Get image dimensions
        num_images = source_images.shape[0]  # N (number of images)
        Image_Z = source_images.shape[1]  # Z (slices per image)
        Image_Y = source_images.shape[2]  # Y (height)
        Image_X = source_images.shape[3]  # X (width)

        f1_col_idx = cellseg_metric.columns.get_loc('F1_Score')
        iou_col_idx = cellseg_metric.columns.get_loc('IoU')

        slice_idx = slice_index  # Start with slice ..
        image_idx = 0
        state = {'image_idx': image_idx, 'slice_idx': slice_idx}

        # Normalize input image
        norm = mcolors.Normalize(vmin=np.percentile(source_images[image_idx, slice_idx], 1), vmax=np.percentile(source_images[image_idx, slice_idx], 99))
        mask_norm = mcolors.Normalize(vmin=0, vmax=1)
        # Set up figure and axes
        fig, axes = plt.subplots(1, 4, figsize=(32, 8))

        # Initialize plots (showing slice slice_index initially)
        im_input = axes[0].imshow(source_images[image_idx, slice_idx], norm=norm, cmap='magma', interpolation='nearest')
        
        # Overlay (Input Image and Prediction Mask) in the second axes
        im_overlay_input = axes[1].imshow(source_images[image_idx, slice_idx],norm=norm, cmap='magma', interpolation='nearest')  # Full opacity for input
        im_overlay_pred = axes[1].imshow(predicted_images[image_idx, slice_idx], norm=mask_norm, alpha=0.5, cmap='Blues')  # Blue prediction mask (with transparency)

        # Display the prediction image in the third axis (full opacity)
        im_pred = axes[2].imshow(predicted_images[image_idx, slice_idx], cmap='Blues', norm=mask_norm, interpolation='nearest')

        # Ground truth in the last axes
        im_gt = axes[3].imshow(ground_truth_images[image_idx, slice_idx], interpolation='nearest', norm=mask_norm, cmap='Greens')

        # Titles
        axes[0].set_title(f'Training source (Image={image_idx}, Z={slice_idx})')
        axes[1].set_title("Overlay: Input + Prediction")
        axes[2].set_title("Prediction")
        axes[3].set_title(f"Ground Truth, F1-Score: {round(cellseg_metric.iloc[image_idx, f1_col_idx], 3)}, IoU: {round(cellseg_metric.iloc[image_idx, iou_col_idx], 3)}")


        print("DEBUG TRAIN DATA TYPE")
        print(source_images[0].dtype)
        print(source_images[0].max)

        for ax in axes:
            ax.axis("off")

        # Add single slider for slice selection
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.02])
        slider = Slider(ax_slider, "Slice", 0, Image_Z - 1, valinit=slice_idx, valstep=1)


        # Function to update plots when slider moves
        def update(val):
            state['slice_idx'] = int(slider.val)
            slice_idx = state['slice_idx']
            image_idx = state['image_idx']

            # Update input image
            im_input.set_data(source_images[image_idx, slice_idx])

            # Update overlay images (input stays the same, prediction updates)
            im_overlay_input.set_data(source_images[image_idx, slice_idx])  # Input image
            im_overlay_pred.set_data(predicted_images[image_idx, slice_idx])  # Prediction mask

            # Update prediction and ground truth images
            im_pred.set_data(predicted_images[image_idx, slice_idx])  
            im_gt.set_data(ground_truth_images[image_idx, slice_idx])  

            axes[0].set_title(f'Training source (Image={image_idx}, Z={slice_idx})')

            # Redraw figure
            fig.canvas.draw_idle()

        # Attach slider to the update function
        slider.on_changed(update)

        # Function to handle text input for both image and slice index
        def on_text_submit(text):

            try:

                # Image selection via TextBox
                image_idx = int(text)
                if image_idx < 0 or image_idx >= num_images:
                    print(f"Invalid image index: {image_idx}. Please enter a value between 0 and {num_images - 1}.")
                    return
                state['image_idx'] = image_idx
                # Trigger image update with current slice_idx to reflect change
                update(slider.val)
                # Update the image display
                im_input.set_data(source_images[image_idx, slider.val])  # Update input image
                im_pred.set_data(predicted_images[image_idx, slider.val])  # Update target image
                im_overlay_input.set_data(source_images[image_idx, slider.val])  # Update input image
                im_overlay_pred.set_data(predicted_images[image_idx, slider.val])  # Update target image
                im_gt.set_data(ground_truth_images[image_idx, slider.val])  # Update target image
                axes[0].set_title(f'Training source (Image={image_idx}, Z={slider.val})')
                axes[3].set_title(f"Ground Truth, F1-Score: {round(cellseg_metric.iloc[image_idx, f1_col_idx], 3)}, IoU: {round(cellseg_metric.iloc[image_idx, iou_col_idx], 3)}")

            except ValueError:
                print("Please enter a valid integer.")

        # Create text boxes for both image and slice index selection
        ax_image_textbox = plt.axes([0.4, 0.1, 0.15, 0.05])  # Positioning of the image textbox
        text_box_image = TextBox(ax_image_textbox, "Image Index:", initial="0")
        text_box_image.on_submit(lambda text: on_text_submit(text))

        plt.show()

# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Config file processing")
parser.add_argument(
    "--config_path", default="./config/fewshot_inference.json", type=str
)
args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

   # Print configuration dictionary pretty
    pprint_config(opt)

    # Run experiment
    main(opt)


