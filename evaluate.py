import numpy as np
import pandas as pd
import argparse
import os
from collections import OrderedDict
from tqdm import tqdm

from train_tools.utils import ConfLoader, pprint_config
from train_tools.measures import evaluate_metrics_cellseg

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
        gt = io.imread(os.path.join(gt_path, name))
        pred = io.imread(os.path.join(pred_path, name))

        # Evaluate metrics
        iou, precision, recall, f1_score = evaluate_metrics_cellseg(pred, gt, threshold=0.5)

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
    # if args.eval_setups.save_path is not None:
    #     os.makedirs(args.eval_setups.save_path, exist_ok=True)
    #     cellseg_metric.to_csv(
    #         os.path.join(args.eval_setups.save_path, "seg_metric.csv"), index=False
    #     )


def show_QC_results(img_path, pred_path, gt_path, cellseg_metric):
    print("now comes the plot")

    source_files = [f for f in os.listdir(img_path) if f.endswith('.tiff') or f.endswith('.tif')]
    prediction_files = [f for f in os.listdir(pred_path) if f.endswith('.tiff') or f.endswith('.tif')]
    target_files = [f for f in os.listdir(gt_path) if f.endswith('.tiff') or f.endswith('.tif')]

    if len(source_files) != len(target_files) or len(source_files) != len(prediction_files):
        raise ValueError("The number of source, prediction, and target files must match.")

    images_list = []
    skipped = 0

    for src_file, pred_file, gt_file in zip(source_files, prediction_files, target_files):
        src_image = io.imread(os.path.join(img_path, src_file))
        pred_image = io.imread(os.path.join(pred_path, pred_file))
        gt_image = io.imread(os.path.join(gt_path, gt_file))

        if src_image.shape != pred_image.shape or src_image.shape != gt_image.shape:
            print(f"[Skipped] Shape mismatch:\n - {src_file}: {src_image.shape}\n - {pred_file}: {pred_image.shape}\n - {gt_file}: {gt_image.shape}")
            skipped += 1
            continue

        images_list.append((src_image, pred_image, gt_image))

    if not images_list:
        raise RuntimeError("No images with matching shapes were found. Cannot continue.")

    if skipped > 0:
        print(f"⚠️ Skipped {skipped} image triplets due to shape mismatches.")

    source_images = np.stack([item[0] for item in images_list])
    predicted_images = np.stack([item[1] for item in images_list])
    ground_truth_images = np.stack([item[2] for item in images_list])

    num_images = source_images.shape[0]
    Image_Z = source_images.shape[1]
    Image_Y = source_images.shape[2]
    Image_X = source_images.shape[3]

    f1_col_idx = cellseg_metric.columns.get_loc('F1_Score')
    iou_col_idx = cellseg_metric.columns.get_loc('IoU')

    slice_idx = 2
    image_idx = 0
    state = {'image_idx': image_idx, 'slice_idx': slice_idx}

    norm = mcolors.Normalize(vmin=np.percentile(source_images[image_idx, slice_idx], 1),
                             vmax=np.percentile(source_images[image_idx, slice_idx], 99))
    mask_norm = mcolors.Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(4, 1, figsize=(50, 15))

    im_input = axes[0].imshow(source_images[image_idx, slice_idx], norm=norm, cmap='magma', interpolation='nearest')
    im_overlay_input = axes[1].imshow(source_images[image_idx, slice_idx], norm=norm, cmap='magma', interpolation='nearest')
    im_overlay_pred = axes[1].imshow(predicted_images[image_idx, slice_idx], norm=mask_norm, alpha=0.5, cmap='Blues')
    im_pred = axes[2].imshow(predicted_images[image_idx, slice_idx], cmap='Blues', norm=mask_norm, interpolation='nearest')
    im_gt = axes[3].imshow(ground_truth_images[image_idx, slice_idx], interpolation='nearest', norm=mask_norm, cmap='Greens')

    axes[0].set_title(f'Training source (Image={image_idx}, Z={slice_idx})')
    axes[1].set_title("Overlay: Input + Prediction")
    axes[2].set_title("Prediction")
    axes[3].set_title(f"Ground Truth, F1-Score: {round(cellseg_metric.iloc[image_idx, f1_col_idx], 3)}, IoU: {round(cellseg_metric.iloc[image_idx, iou_col_idx], 3)}")

    for ax in axes:
        ax.axis("off")

    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.02])
    slider = Slider(ax_slider, "Slice", 0, Image_Z - 1, valinit=slice_idx, valstep=1)

    def update(val):
        state['slice_idx'] = int(slider.val)
        slice_idx = state['slice_idx']
        image_idx = state['image_idx']

        im_input.set_data(source_images[image_idx, slice_idx])
        im_overlay_input.set_data(source_images[image_idx, slice_idx])
        im_overlay_pred.set_data(predicted_images[image_idx, slice_idx])
        im_pred.set_data(predicted_images[image_idx, slice_idx])
        im_gt.set_data(ground_truth_images[image_idx, slice_idx])
        axes[0].set_title(f'Training source (Image={image_idx}, Z={slice_idx})')

        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_text_submit(text):
        try:
            image_idx = int(text)
            if image_idx < 0 or image_idx >= num_images:
                print(f"Invalid image index: {image_idx}. Please enter a value between 0 and {num_images - 1}.")
                return
            state['image_idx'] = image_idx
            update(slider.val)
            im_input.set_data(source_images[image_idx, slider.val])
            im_pred.set_data(predicted_images[image_idx, slider.val])
            im_overlay_input.set_data(source_images[image_idx, slider.val])
            im_overlay_pred.set_data(predicted_images[image_idx, slider.val])
            im_gt.set_data(ground_truth_images[image_idx, slider.val])
            axes[0].set_title(f'Training source (Image={image_idx}, Z={slider.val})')
            axes[3].set_title(f"Ground Truth, F1-Score: {round(cellseg_metric.iloc[image_idx, f1_col_idx], 3)}, IoU: {round(cellseg_metric.iloc[image_idx, iou_col_idx], 3)}")
        except ValueError:
            print("Please enter a valid integer.")

    ax_image_textbox = plt.axes([0.4, 0.1, 0.15, 0.05])
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
