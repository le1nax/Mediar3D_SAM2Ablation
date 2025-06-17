import os, glob
import json
import argparse
from skimage import io
import numpy as np


def official_paths_labeled(image_path, label_path):
    """Map paths for official labeled datasets as dictionary list
    Only includes images where the corresponding label has at least one non-zero pixel.
    """
    image_path = os.path.join(image_path, "*")
    label_path = os.path.join(label_path, "*")

    images_raw = sorted(glob.glob(image_path))
    labels_raw = sorted(glob.glob(label_path))

    # Ensure equal length pairing
    images_raw = images_raw[:len(labels_raw)]

    data_dicts = []

    for img_path, lb_path in zip(images_raw, labels_raw):
        label_image = io.imread(lb_path)

        # Check if there's at least one non-zero pixel
        if np.any(label_image):
            data_item = {
                "img": img_path,
                "label": lb_path,
            }
            data_dicts.append(data_item)

    map_dict = {"official": data_dicts}

    return map_dict


def official_paths_tuning(image_path):
    """Map paths for official tuning datasets as dictionary list"""
    image_path = os.path.join(image_path, "*")
    images_raw = sorted(glob.glob(image_path))

    data_dicts = []

    for img_path in images_raw:
        data_item = {"img": img_path}
        data_dicts.append(data_item)

    map_dict = {"official": data_dicts}

    return map_dict


def add_mapping_to_json(json_file, map_dict):
    """Save mapped dictionary as a json file"""

    if not os.path.exists(json_file):
        with open(json_file, "w") as file:
            json.dump({}, file)

    with open(json_file, "r") as file:
        data = json.load(file)

    for map_key, map_item in map_dict.items():
        if map_key not in data.keys():
            data[map_key] = map_item
        else:
            print('>>> "{}" already exists in path map keys...'.format(map_key))

    with open(json_file, "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    # [!Caution] The paths should be overrided for the local environment!
    parser = argparse.ArgumentParser(description="Mapping files and paths")
    parser.add_argument("--pred_path", default="/work/scratch/geiger/Datasets/CTC/sim3d/01_2d", type=str)
    parser.add_argument("--train_img_path", default="/work/scratch/geiger/Datasets/CTC/sim3d/01_2d", type=str)
    parser.add_argument("--train_label_path", default="/work/scratch/geiger/Datasets/CTC/sim3d/01_GT_2d", type=str)
    parser.add_argument("--data", default="dic_sim", type=str)
    args = parser.parse_args()

    MAP_DIR = "./train_tools/data_utils/"

    print("\n----------- Path Mapping for Tuning Data is Started... -----------\n")

    map_labeled = os.path.join(MAP_DIR, f"mapping_tuning_{args.data}.json")
    map_dict = official_paths_tuning(args.pred_path)
    add_mapping_to_json(map_labeled, map_dict)


    print("\n----------- Path Mapping for Labeled Data is Started... -----------\n")

    map_labeled = os.path.join(MAP_DIR, f"mapping_labeled_{args.data}.json")
    map_dict = official_paths_labeled(args.train_img_path, args.train_label_path)
    add_mapping_to_json(map_labeled, map_dict)
