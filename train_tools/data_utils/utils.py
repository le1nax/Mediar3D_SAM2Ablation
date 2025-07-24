import os
import json
import numpy as np

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