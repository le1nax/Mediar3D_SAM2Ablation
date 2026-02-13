import os
import re
import glob
import csv
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from train_tools import *
from SetupDict import MODELS, PREDICTOR

from train_tools.measures import evaluate_metrics_cellseg, compute_CTC_SEG_fast, average_precision_final
import tifffile as tif

def round_to_magnitude(n):
    """Round ROI number to order of magnitude (0, 10, 100, 1000, 10000, ...)"""
    if n == 0:
        return 0
    magnitude = 10 ** int(np.floor(np.log10(n)))
    return int(np.floor(n / magnitude) * magnitude)


def parse_roi_group(model_name):
    """Extract ROI number from model filename and round to order of magnitude"""
    match = re.search(r"_ROI_(\d+)_", model_name)
    if match:
        roi_num = int(match.group(1))
        return round_to_magnitude(roi_num)
    raise RuntimeError(f"no ROI number given for model: {model_name}")


def run_grid_prediction(config_path):
    # =====================================================
    # 1. Load Base Config
    # =====================================================
    opt = ConfLoader(config_path).opt
    setups = opt.pred_setups
    pprint_config(opt)

    model_dir = setups.model_path
    thresholds = setups.algo_params["cellprob_thresholds"]
    input_path = setups.input_path
    gt_path = setups.ground_truth_path
    device = setups.device

    dataset_name = os.path.basename(os.path.normpath(input_path))
    output_root = setups.output_path

    # =====================================================
    # 2. Find all Models
    # =====================================================
    model_paths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    print(f"Found {len(model_paths)} checkpoints in {model_dir}")

    # =====================================================
    # 3. Run Grid Prediction
    # =====================================================
    for model_path in model_paths:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        roi_group = parse_roi_group(model_name)
        print(f"\nEvaluating model: {model_name} | ROI group: {roi_group}")

        # Load model
        model_args = setups.model
        model = MODELS[model_args.name](**model_args.params)
        weights = torch.load(model_path, map_location="cpu")
        model.load_state_dict(weights, strict=False)
        model.to(device)
        model.eval()

        # Predictor object (fixed)
        predictor = PREDICTOR[setups.name](
            model,
            device,
            input_path,
            None,  # output_path set dynamically below
            "",
            setups.make_submission,
            setups.exp_name,
            setups.algo_params,
        )

        for th in thresholds:
            print(f"  → Threshold: {th}")

            # Construct output folder tree
            th_folder = f"th{th}"
            out_dir = os.path.join(output_root, dataset_name, model_name, f"ROI_{roi_group}", th_folder)
            os.makedirs(out_dir, exist_ok=True)
            predictor.output_path = out_dir

            # Run prediction
            predictor.cellprob_threshold = th
            predictor.conduct_prediction()

            # Prepare CSV path for this threshold
            csv_path = os.path.join(out_dir, f"metrics_{model_name}_ROI{roi_group}_th{th}.csv")
            header = ["image_name", "SEG", "IoU", "AP05"]

            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

            #Evaluate predictions
            pred_files = sorted(glob.glob(os.path.join(out_dir, "*.tif*")))

            for pred_path in tqdm(pred_files, desc=f"Evaluating {model_name} @ {th}"):
                image_name = os.path.basename(pred_path)
                gt_file = os.path.join(gt_path, image_name)
                if not os.path.exists(gt_file):
                    print(f"Skipping {image_name} (GT missing)")
                    continue

                pred_mask = tif.imread(pred_path)
                gt_mask = tif.imread(gt_file)

                seg_score, _ = compute_CTC_SEG_fast(gt_mask, pred_mask)
                AP05_score = average_precision_final(gt_mask, pred_mask)
                iou,_,_1,_2 = evaluate_metrics_cellseg(gt_mask, pred_mask)

                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([image_name, seg_score, iou, AP05_score])

            print(f"Metrics saved: {csv_path}")

    print("\nll grid predictions completed.")


if __name__ == "__main__":
    CONFIG_PATH = "./config/step3_prediction/grid_predictions.json"
    run_grid_prediction(CONFIG_PATH)



# ################################THIS VERSION EXTRACTS RESULTS FOR 2D LABELS
# import os
# import re
# import glob
# import csv
# import torch
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime

# from train_tools import *
# from SetupDict import MODELS, PREDICTOR
# from train_tools.measures import evaluate_metrics_cellseg, compute_CTC_SEG_fast, average_precision_final
# import tifffile as tif


# def round_to_magnitude(n):
#     """Round ROI number to order of magnitude (0, 10, 100, 1000, 10000, ...)"""
#     if n == 0:
#         return 0
#     magnitude = 10 ** int(np.floor(np.log10(n)))
#     return int(np.floor(n / magnitude) * magnitude)


# def parse_roi_group(model_name):
#     """Extract ROI number from model filename and round to order of magnitude"""
#     match = re.search(r"_ROI_(\d+)_", model_name)
#     if match:
#         roi_num = int(match.group(1))
#         return round_to_magnitude(roi_num)
#     raise RuntimeError(f"no ROI number given for model: {model_name}")


# def run_grid_prediction(config_path):
#     # =====================================================
#     # 1. Load Base Config
#     # =====================================================
#     opt = ConfLoader(config_path).opt
#     setups = opt.pred_setups
#     pprint_config(opt)

#     model_dir = setups.model_path
#     thresholds = setups.algo_params["cellprob_thresholds"]
#     input_path = setups.input_path
#     gt_path = setups.ground_truth_path
#     device = setups.device

#     dataset_name = os.path.basename(os.path.normpath(input_path))
#     output_root = setups.output_path

#     # =====================================================
#     # 2. Find all Models
#     # =====================================================
#     model_paths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
#     print(f"🔍 Found {len(model_paths)} checkpoints in {model_dir}")

#     # =====================================================
#     # 3. Run Grid Prediction
#     # =====================================================
#     for model_path in model_paths:
#         model_name = os.path.splitext(os.path.basename(model_path))[0]
#         roi_group = parse_roi_group(model_name)
#         print(f"\n🧠 Evaluating model: {model_name} | ROI group: {roi_group}")

#         # Load model
#         model_args = setups.model
#         model = MODELS[model_args.name](**model_args.params)
#         weights = torch.load(model_path, map_location="cpu")
#         model.load_state_dict(weights, strict=False)
#         model.to(device)
#         model.eval()

#         # Predictor
#         predictor = PREDICTOR[setups.name](
#             model,
#             device,
#             input_path,
#             None,  # output_path will be updated below
#             "",
#             setups.make_submission,
#             setups.exp_name,
#             setups.algo_params,
#         )

#         # =====================================================
#         # Threshold loop
#         # =====================================================
#         for th in thresholds:
#             print(f"  → Threshold: {th}")

#             # Create folder tree
#             th_folder = f"th{th}"
#             out_dir = os.path.join(output_root, dataset_name, model_name, f"ROI_{roi_group}", th_folder)
#             os.makedirs(out_dir, exist_ok=True)
#             predictor.output_path = out_dir

#             # Run predictions
#             predictor.cellprob_threshold = th
#             predictor.conduct_prediction()

#             # Prepare CSV for metrics
#             csv_path = os.path.join(out_dir, f"metrics_{model_name}_ROI{roi_group}_th{th}.csv")
#             header = ["image_name", "timestamp", "zslice", "SEG", "IoU", "AP05"]
#             with open(csv_path, "w", newline="") as f:
#                 csv.writer(f).writerow(header)

#             # =====================================================
#             # Evaluate 2D GTs vs 3D predictions
#             # =====================================================
#             pred_files = {os.path.basename(p): p for p in sorted(glob.glob(os.path.join(out_dir, "t*_label.tif*")))}
#             gt_files = sorted(glob.glob(os.path.join(gt_path, "man_seg_*.tif*")))

#             for gt_file in tqdm(gt_files, desc=f"Evaluating {model_name} @ {th}"):
#                 gt_name = os.path.basename(gt_file)
#                 match = re.match(r"man_seg_(\d{3})_(\d{3})\.tif", gt_name)
#                 if not match:
#                     print(f"⚠️ Skipping {gt_name}: invalid name pattern")
#                     continue

#                 t_str, z_str = match.groups()
#                 t_idx, z_idx = int(t_str), int(z_str)

#                 # Corresponding 3D prediction file
#                 pred_name = f"t{t_str}_label.tiff"
#                 if pred_name not in pred_files:
#                     print(f"⚠️ Missing prediction for time {t_str}")
#                     continue

#                 # Load prediction volume
#                 pred_vol = tif.imread(pred_files[pred_name])
#                 if z_idx >= pred_vol.shape[0]:
#                     print(f"⚠️ z={z_idx} out of range for {pred_name}")
#                     continue

#                 pred_slice = pred_vol[z_idx]
#                 gt_mask = tif.imread(gt_file)

#                 # Metrics
#                 seg_score, _ = compute_CTC_SEG_fast(gt_mask, pred_slice)
#                 AP05_score = average_precision_final(gt_mask, pred_slice)
#                 iou, _, _, _ = evaluate_metrics_cellseg(gt_mask, pred_slice)

#                 with open(csv_path, "a", newline="") as f:
#                     csv.writer(f).writerow([gt_name, t_idx, z_idx, seg_score, iou, AP05_score])

#             print(f"✅ Metrics saved: {csv_path}")

#     print("\n🎯 All grid predictions completed.")


# if __name__ == "__main__":
#     CONFIG_PATH = "./config/step3_prediction/grid_predictions.json"
#     run_grid_prediction(CONFIG_PATH)
