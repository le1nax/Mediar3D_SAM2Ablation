import sys, os
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import tifffile

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.MEDIAR.utils import *

__all__ = ["split_train_valid", "path_decoder"]


def split_train_valid(data_dicts, valid_portion=0.1):
    """Split train/validata data according to the given proportion"""

    train_dicts, valid_dicts = data_dicts, []
    if valid_portion > 0:

        # Obtain & shuffle data indices
        num_data_dicts = len(data_dicts)
        indices = np.arange(num_data_dicts)
        np.random.shuffle(indices)

        # Divide train/valid indices by the proportion
        valid_size = int(num_data_dicts * valid_portion)
        train_indices = indices[valid_size:]
        valid_indices = indices[:valid_size]

        # Assign data dicts by split indices
        train_dicts = [data_dicts[idx] for idx in train_indices]
        valid_dicts = [data_dicts[idx] for idx in valid_indices]

    print(
        "\n(DataLoaded) Training data size: %d, Validation data size: %d\n"
        % (len(train_dicts), len(valid_dicts))
    )

    return train_dicts, valid_dicts


def path_decoder(root, mapping_file, no_label=False, unlabeled=False):
    """Decode img/label/cellcenter file paths from root & mapping directory.

    Args:
        root (str): Base path for dataset
        mapping_file (str): JSON file containing image & label file paths
        no_label (bool): If True, do not include labels in output
        unlabeled (bool): If True, exclude certain corrupted images

    Returns:
        list[dict]: list of dictionaries (with keys "img", "label", optionally "cellcenter")
    """
    data_dicts = []

    with open(mapping_file, "r") as file:
        data = json.load(file)

        for map_key in data.keys():
            data_dict_item = []

            for elem in data[map_key]:
                item = {"img": os.path.join(root, elem["img"])}

                if not no_label and "label" in elem:
                    item["label"] = os.path.join(root, elem["label"])

                if "cellcenter" in elem:
                    item["cellcenter"] = os.path.join(root, elem["cellcenter"])

                data_dict_item.append(item)

            data_dicts += data_dict_item

    if unlabeled:
        data_dicts = [d for d in data_dicts if "00504" not in d["img"]]

    return data_dicts

def add_flows(data_dicts, device="cuda", overwrite=False, save_as_tiff=True, precompute_flows=False):
    """
    Adds or computes flow paths based on label paths.

    Args:
        data_dicts (list[dict]): list of samples from path_decoder
        device (str): device to run flow computation
        overwrite (bool): force recompute even if exists
        save_as_tiff (bool): save flow as .tiff instead of .npy

    Returns:
        list[dict]: modified list with "flow" key added
    """
    for data in tqdm(data_dicts, desc="Checking or generating flows"):
        if "label" not in data:
            continue

        if not precompute_flows:
            data["flow"] = None
            continue

        # Create "flows" folder inside the labels directory
        label_path = Path(data["label"])
        labels_dir = label_path.parent  # folder containing the label
        flow_dir = labels_dir / "flows"  # flows subfolder inside labels

        flow_dir.mkdir(parents=True, exist_ok=True)

        stem = label_path.stem.replace("_label", "")
        flow_suffix = "_flow.tiff" if save_as_tiff else "_flow.npy"
        flow_path = flow_dir / f"{stem}{flow_suffix}"

        if not flow_path.exists() or overwrite:
            label_img = tifffile.imread(label_path).astype(np.int32)
            label_tensor = torch.tensor(label_img, dtype=torch.int32, device=device).unsqueeze(0)
            flows = labels_to_flows(label_tensor, use_gpu=True, device=device)
            flow_np = flows.squeeze(0)

            if save_as_tiff:
                flow_np = np.moveaxis(flow_np, 0, -1)
                tifffile.imwrite(flow_path, flow_np.astype(np.float32))
            else:
                np.save(flow_path, flow_np)

        data["flow"] = str(flow_path)

    return data_dicts