import numpy as np
import pandas as pd
import tifffile as tif
import argparse
import os
from collections import OrderedDict
from tqdm import tqdm

from train_tools.utils import ConfLoader, pprint_config
from train_tools.measures import evaluate_f1_score_cellseg, evaluate_f1_score_cellseg_edited

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from matplotlib.widgets import TextBox
from skimage import io
import pandas as pd


def main(args):
    
    # Get files from the paths
    gt_path, pred_path, img_path = args.eval_setups.gt_path, args.eval_setups.pred_path, args.eval_setups.img_path
    names = sorted(os.listdir(pred_path))

    names_total = []
    ious_total, precisions_total, recalls_total, f1_scores_total = [], [], [], []

    for name in tqdm(names):
        assert name.endswith("_label.tiff"), "The suffix of label name should be _label.tiff"

        # Load images
        gt = tif.imread(os.path.join(gt_path, name))
        pred = tif.imread(os.path.join(pred_path, name))

        # Evaluate metrics
        iou, precision, recall, f1_score = evaluate_f1_score_cellseg_edited(gt, pred, threshold=0.5)

        names_total.append(name)
        ious_total.append(np.round(iou, 4))
        precisions_total.append(np.round(precision, 4))
        recalls_total.append(np.round(recall, 4))
        f1_scores_total.append(np.round(f1_score, 4))

    # Compile results into DataFrame
    cellseg_metric = OrderedDict()
    cellseg_metric["Names"] = names_total
    cellseg_metric["IoU"] = ious_total
    cellseg_metric["Precision"] = precisions_total
    cellseg_metric["Recall"] = recalls_total
    cellseg_metric["F1_Score"] = f1_scores_total

    cellseg_metric = pd.DataFrame(cellseg_metric)

    # Show results
    print("mean IoU:", np.mean(cellseg_metric["IoU"]))
    print("mean F1 Score:", np.mean(cellseg_metric["F1_Score"]))
        

    ###########Vis

    source_files = [f for f in os.listdir(pred_path) if f.endswith('.tiff') or f.endswith('.tif')]
    target_files = [f for f in os.listdir(gt_path) if f.endswith('.tiff') or f.endswith('.tif')]

    if len(source_files) != len(target_files):
        raise ValueError("The number of source and target files does not match.")

    # Initialize arrays to hold the images
    images_list = []

    # Load all the images from the source and target directories
    for src_file, tgt_file in zip(source_files, target_files):
        src_image = io.imread(os.path.join(pred_path, src_file))
        tgt_image = io.imread(os.path.join(gt_path, tgt_file))
        
        images_list.append((src_image, tgt_image))


    show_QC_results(img_path, pred_path, gt_path, cellseg_metric)

    # Save results
    if args.eval_setups.save_path is not None:
        os.makedirs(args.eval_setups.save_path, exist_ok=True)
        cellseg_metric.to_csv(
            os.path.join(args.eval_setups.save_path, "seg_metric.csv"), index=False
        )


def show_QC_results(img_path, pred_path, gt_path, cellseg_metric):
        print("now comes the plot")

        source_files = [f for f in os.listdir(img_path) if f.endswith('.tiff') or f.endswith('.tif')]
        prediction_files = [f for f in os.listdir(pred_path) if f.endswith('.tiff') or f.endswith('.tif')]
        target_files = [f for f in os.listdir(gt_path) if f.endswith('.tiff') or f.endswith('.tif')]

        if len(source_files) != len(target_files) or len(source_files) != len(prediction_files):
            raise ValueError("The number of source, prediction and target files does not match.")

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

        slice_idx = 25  # Start with slice ..
        image_idx = 0
        state = {'image_idx': image_idx, 'slice_idx': slice_idx}

        # Normalize input image
        norm = mcolors.Normalize(vmin=np.percentile(source_images[image_idx, slice_idx], 1), vmax=np.percentile(source_images[image_idx, slice_idx], 99))
        mask_norm = mcolors.Normalize(vmin=0, vmax=1)
        # Set up figure and axes
        fig, axes = plt.subplots(1, 4, figsize=(32, 8))

        # Initialize plots (showing slice 25 initially)
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
    "--config_path", default="./config/eval_conf.json", type=str
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
