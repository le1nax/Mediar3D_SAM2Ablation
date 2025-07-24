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


def main(args):

    slice_index = args.eval_setups.slice_index

    # Output directory to save slices
    output_img_dir = args.eval_setups.fs_train_img
    output_val_dir = args.eval_setups.fs_train_masks
    output_cellcenter_dir = args.eval_setups.fs_train_cellcenters

    # os.makedirs(output_img_dir, exist_ok=True)
    # os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_cellcenter_dir, exist_ok=True)

    # Output directory to save inference image
    output_inference_img_dir = args.eval_setups.fs_inference_img
    output_inference_val_dir = args.eval_setups.fs_inference_masks
    output_inference_cellcenters_dir = args.eval_setups.fs_inference_cellcenters

    # os.makedirs(output_img_dir, exist_ok=True)
    # os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_inference_cellcenters_dir, exist_ok=True)


    
    # Get files from the paths
    gt_path, pred_path, img_path, cellcenters_path = args.eval_setups.gt_path, args.eval_setups.pred_path, args.eval_setups.img_path, args.eval_setups.cellcenter_path
    img_names = sorted(os.listdir(img_path))
    gt_names = sorted(os.listdir(gt_path))
    cellcenter_names = sorted(os.listdir(cellcenters_path))


    for i in tqdm(range(len(img_names))):

        # Load images
        gt = tif.imread(os.path.join(gt_path, gt_names[i]))
        img = tif.imread(os.path.join(img_path, img_names[i]))
        cellcenters = tif.imread(os.path.join(cellcenters_path, cellcenter_names[i]))
        

        ###slice 2d train images
        img_crop = img[:slice_index]
        gt_crop = gt[:slice_index]
        cellcenters_proj = z_project_cellcenters(cellcenters)
        cellcenters_proj_crop = cellcenters_proj[:slice_index]
        

        for i, slice_2d in enumerate(img_crop):
            filename = f"cell_{i:05d}.tiff"  # zero-padded to 5 digits
            filepath = os.path.join(output_img_dir, filename)
            tif.imwrite(filepath, slice_2d)
            print(f"Saved {len(img_crop)} slices to '{output_img_dir}' directory.")

        for i, slice_2d in enumerate(gt_crop):
            filename = f"cell_{i:05d}_label.tiff"  # zero-padded to 5 digits
            filepath = os.path.join(output_val_dir, filename)
            tif.imwrite(filepath, slice_2d)
            print(f"Saved {len(gt_crop)} slices to '{output_val_dir}' directory.")

        for i, slice_2d in enumerate(cellcenters_proj_crop):
            filename = f"cell_centers_{i:05d}.tiff"  # zero-padded to 5 digits
            filepath = os.path.join(output_cellcenter_dir, filename)
            tif.imwrite(filepath, slice_2d)
            print(f"Saved {len(cellcenters_proj_crop)} slices to '{output_cellcenter_dir}' directory.")
        
        ###crop 3d few shot inference image

    #     img_infer = img[slice_index:]
    #     gt_infer = gt[slice_index:]
    #     cellcenters_infer = cellcenters[slice_index:]

    #     infer_img_path = os.path.join(output_inference_img_dir, f"infer_{i:03d}.tiff")
    #     infer_gt_path = os.path.join(output_inference_val_dir, f"infer_{i:03d}_label.tiff")
    #     infer_cellcenter_path = os.path.join(output_inference_cellcenters_dir, f"infer_{i:03d}_cellcenter.tiff")

    #     tif.imwrite(infer_img_path, img_infer)
    #     tif.imwrite(infer_gt_path, gt_infer)
    #     tif.imwrite(infer_cellcenter_path, cellcenters_infer)


    # print(f"Saved inference volume to '{infer_img_path}' and '{infer_gt_path}'.")


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


