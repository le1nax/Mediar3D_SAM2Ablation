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
            "../../Datasets/CTC/sim3d/01",
            "../../Datasets/pretraining_mediar/CellposeDataset/01",
            "../../Datasets/pretraining_mediar/DataScienceBowl/01_tiff",
            "../../Datasets/pretraining_mediar/livecell_A172/01",
            "../../Datasets/pretraining_mediar/livecell_BT474/01",
            "../../Datasets/pretraining_mediar/livecell_BV2/01",
            "../../Datasets/pretraining_mediar/livecell_Huh7/01",
            "../../Datasets/pretraining_mediar/livecell_MCF7/01",
            "../../Datasets/pretraining_mediar/livecell_SHSY5Y/01",
            "../../Datasets/pretraining_mediar/livecell_SkBr3/01",
            "../../Datasets/pretraining_mediar/livecell_SKOV3/01",
            "../../Datasets/pretraining_mediar/OmniposeDataset/01"
        ],
        help="List of image directories"
    )
    parser.add_argument(
        "--train_label_paths",
        nargs="+",
        default=[
            "../../Datasets/CTC/sim3d/01_GT",
            "../../Datasets/pretraining_mediar/CellposeDataset/01_ST/SEG",
            "../../Datasets/pretraining_mediar/DataScienceBowl/01_GT_tiff/SEG",
            "../../Datasets/pretraining_mediar/livecell_A172/01_ST/SEG",
            "../../Datasets/pretraining_mediar/livecell_BT474/01_ST/SEG",
            "../../Datasets/pretraining_mediar/livecell_BV2/01_ST/SEG",
            "../../Datasets/pretraining_mediar/livecell_Huh7/01_ST/SEG",
            "../../Datasets/pretraining_mediar/livecell_MCF7/01_ST/SEG",
            "../../Datasets/pretraining_mediar/livecell_SHSY5Y/01_ST/SEG",
            "../../Datasets/pretraining_mediar/livecell_SkBr3/01_ST/SEG",
            "../../Datasets/pretraining_mediar/livecell_SKOV3/01_ST/SEG",
            "../../Datasets/pretraining_mediar/OmniposeDataset/01_ST/SEG"
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