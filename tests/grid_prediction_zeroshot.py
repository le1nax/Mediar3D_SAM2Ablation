import os
import glob
import csv
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from train_tools import *
from SetupDict import MODELS, PREDICTOR
from train_tools.measures import evaluate_metrics_cellseg, compute_CTC_SEG_fast, average_precision_final, dice_per_annotated_cell_3d
import tifffile as tif


def run_multi_threshold_prediction(config_path):
    # =====================================================
    # 1. Load Base Config
    # =====================================================
    opt = ConfLoader(config_path).opt
    setups = opt.pred_setups
    pprint_config(opt)

    model_path = setups.model_path  # now a single model file, not a folder
    thresholds = setups.algo_params["cellprob_thresholds"]
    input_path = setups.input_path
    gt_path = setups.ground_truth_path
    device = setups.device
    output_root = setups.output_path

    dataset_name = os.path.basename(os.path.normpath(input_path))
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    print(f"\n🧠 Using model: {model_name}")

    # =====================================================
    # 2. Load Model
    # =====================================================
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
        None,  # output path will be set dynamically below
        "",
        setups.make_submission,
        setups.exp_name,
        setups.algo_params,
    )

    # =====================================================
    # 3. Run Predictions for Each Threshold
    # =====================================================
    for th in thresholds:
        print(f"\n→ Threshold: {th}")

        # Construct output folder
        th_folder = f"th{th}"
        out_dir = os.path.join(output_root, dataset_name, model_name, th_folder)
        os.makedirs(out_dir, exist_ok=True)
        predictor.output_path = out_dir

        # Run prediction
        predictor.cellprob_threshold = th
        predictor.conduct_prediction()

        # Prepare CSV
        csv_path = os.path.join(out_dir, f"metrics_{model_name}_th{th}.csv")
        header = ["image_name", "SEG"]

        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

        # Evaluate predictions
        pred_files = sorted(glob.glob(os.path.join(out_dir, "*.tif*")))
        for pred_path in tqdm(pred_files, desc=f"Evaluating {model_name} @ th={th}"):
            image_name = os.path.basename(pred_path)
            gt_file = os.path.join(gt_path, image_name)

            if not os.path.exists(gt_file):
                print(f"⚠️ Skipping {image_name} (GT missing)")
                continue

            pred_mask = tif.imread(pred_path)
            gt_mask = tif.imread(gt_file)

            seg_score, _ = compute_CTC_SEG_fast(gt_mask, pred_mask)
            #dice_score, _ = dice_per_annotated_cell_3d(pred_mask, gt_mask)


            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([image_name, seg_score])

        print(f"✅ Metrics saved: {csv_path}")

    print("\n🎯 All threshold predictions completed.")


if __name__ == "__main__":
    CONFIG_PATH = "./config/step3_prediction/grid_predictions.json"
    run_multi_threshold_prediction(CONFIG_PATH)
