import os, glob
import json
import argparse


def official_paths_labeled(image_path, label_path, center_path=None):
    """Map paths for official labeled datasets as dictionary list
    Only includes images where the corresponding label has at least one non-zero pixel.
    """
    image_path = os.path.join(image_path, "*")
    label_path = os.path.join(label_path, "*")
    center_path = os.path.join(center_path, "*") if center_path is not None else None

    images_raw = sorted(glob.glob(image_path))
    labels_raw = sorted(glob.glob(label_path))
    centers_raw = sorted(glob.glob(center_path)) if center_path is not None else [None] * len(images_raw)

    # Ensure equal length pairing
    min_len = min(len(images_raw), len(labels_raw), len(centers_raw))
    images_raw = images_raw[:min_len]
    labels_raw = labels_raw[:min_len]
    centers_raw = centers_raw[:min_len]

    data_dicts = []
    for img_path, lb_path, ct_path in zip(images_raw, labels_raw, centers_raw):
        data_item = {"img": img_path, "label": lb_path}
        if ct_path is not None:
            data_item["cellcenter"] = ct_path
        data_dicts.append(data_item)

    return data_dicts


def official_paths_tuning(image_path):
    """Map paths for official tuning datasets as dictionary list"""
    image_path = os.path.join(image_path, "*")
    images_raw = sorted(glob.glob(image_path))
    data_dicts = [{"img": img_path} for img_path in images_raw]
    return data_dicts


def add_mapping_to_json(json_file, map_dict):
    """Save mapped dictionary as a json file"""
    if os.path.exists(json_file):
        os.remove(json_file)
        print(f'>>> Removed existing mapping file: {json_file}')

    with open(json_file, "w") as file:
        json.dump(map_dict, file, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mapping files and paths")
    parser.add_argument("--pred_path", default="../../Datasets/dummy_dataset/01_img", type=str)

    # multiple datasets for training
    parser.add_argument(
        "--train_img_paths",
        nargs="+",
        default=[
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/Cellpose_Data/train_CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/DSBowl2018/data-science-bowl-2018/stage1_train/CTC_format/01_tiff",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/A172/CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/BT474/CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/BV2/CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/Huh7/CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/MCF7/CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/SHSY5Y/CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/SkBr3/CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/SKOV3/CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/omnipose/datasets/CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/BCCD_test_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/BCCD_train_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cellpose_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cellpose_test_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/CoNIC_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cpm15_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cpm17_test_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cpm17_train_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cyto2_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/DeepBacs_test_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/DeepBacs_train_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/IHC_TMA_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/lynsec_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/neurips_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/nuinsseg_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/panNuke2_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/panNuke_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_test_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_train_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_val_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/TNBC_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/yeast_BF_img",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/yeast_PhC_img"

        ],
        help="List of image directories"
    )
    
    
    parser.add_argument(
        "--train_label_paths",
        nargs="+",
        default=[
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/Cellpose_Data/train_CTC_format/01_ST/SEG",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/DSBowl2018/data-science-bowl-2018/stage1_train/CTC_format/01_GT_tiff/SEG",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/A172/CTC_format/01_ST/SEG",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/BT474/CTC_format/01_ST/SEG",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/BV2/CTC_format/01",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/Huh7/CTC_format/01_ST/SEG",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/MCF7/CTC_format/01_ST/SEG",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/SHSY5Y/CTC_format/01_ST/SEG",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/SkBr3/CTC_format/01_ST/SEG",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/SKOV3/CTC_format/01_ST/SEG",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/omnipose/datasets/CTC_format/01_ST/SEG",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/BCCD_test_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/BCCD_train_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cellpose_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cellpose_test_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/CoNIC_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cpm15_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cpm17_test_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cpm17_train_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/cyto2_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/DeepBacs_test_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/DeepBacs_train_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/IHC_TMA_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/lynsec_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/neurips_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/nuinsseg_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/panNuke2_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/panNuke_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_test_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_train_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_val_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/TNBC_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/yeast_BF_masks",
            "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/yeast_PhC_masks"
        ],
        help="List of label directories"
    )
    parser.add_argument("--data", default="dic_sim", type=str)

    args = parser.parse_args()
    


    MAP_DIR = "./train_tools/data_utils/"
    os.makedirs(MAP_DIR, exist_ok=True)

    print("\n----------- Path Mapping for Tuning Data is Started... -----------\n")
    map_tuning = {"official": official_paths_tuning(args.pred_path)}
    add_mapping_to_json(os.path.join(MAP_DIR, f"mapping_tuning_{args.data}.json"), map_tuning)

    print("\n----------- Path Mapping for Labeled Data is Started... -----------\n")

    if len(args.train_img_paths) != len(args.train_label_paths):
        raise ValueError("Number of train_img_paths and train_label_paths must match!")

    all_data_dicts = []
    for img_dir, label_dir in zip(args.train_img_paths, args.train_label_paths):
        all_data_dicts.extend(official_paths_labeled(img_dir, label_dir))

    map_labeled = {"official": all_data_dicts}
    add_mapping_to_json(os.path.join(MAP_DIR, f"mapping_labeled_{args.data}.json"), map_labeled)

    # --- After creating all_data_dicts ---
    dataset_names = set()
    for item in all_data_dicts:
        # take the parent folder name of the image path
        dataset_name = os.path.basename(os.path.dirname(item["img"]))
        dataset_names.add(dataset_name)

    dataset_names = sorted(list(dataset_names))

    print("\nDetected datasets from mapping file:")
    for name in dataset_names:
        print(f"  '{name}': <YOUR_RATIO>,")
    print("\n# Copy-paste the above into your custom_ratios dict and fill in probabilities.\n")
