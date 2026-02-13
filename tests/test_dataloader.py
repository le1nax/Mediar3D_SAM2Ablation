import os
import numpy as np
import tifffile as tif
from skimage import io
from monai.data import Dataset, DataLoader
from pathlib import Path
import tifffile
import json
import torch


from train_tools.data_utils.transforms import train_transforms, valid_transforms

class CustomMediarDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__(data, transform)

    def __getitem__(self, index):
        data = dict(self.data[index])  # Make a copy

        # Handle cellcenter paths as before
        cellcenter_path = data.get("cellcenter", None)
        if cellcenter_path is not None:
            cellcenter_path = Path(cellcenter_path)
            if not cellcenter_path.exists():
                print(f"[Warning] cellcenter file not found at index {index}: {cellcenter_path}")
            else:
                data["cellcenter"] = str(cellcenter_path)
        else:
            data.pop("cellcenter", None)

        # --- New: load precomputed flow ---
        flow_path = data.get("flow", None)
        if flow_path is not None:
            flow_path = Path(flow_path)
            if not flow_path.exists():
                print(f"[Warning] flow file not found at index {index}: {flow_path}")
                data.pop("flow", None)
            else:
                # Load flow data (assuming numpy .npy, adjust if different)
                flow_np = tifffile.imread(flow_path)
                flow_tensor = torch.from_numpy(flow_np).float()
                data["flow"] = flow_tensor
        else:
            data.pop("flow", None)
            
        if self.transform:
            data = self.transform(data)

        return data
    

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

def get_dataloaders_labeled(
    root,
    mapping_file,
    batch_size=1,
):
    """Set DataLoaders for labeled datasets.

    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
        shuffle (bool, optional): shuffles dataloader. Defaults to True.
        num_workers (int, optional): number of workers for each datalaoder. Defaults to 5.

    Returns:
        dict: dictionary of data loaders.
    """

    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file)


    # Obtain datasets with transforms
    trainset = CustomMediarDataset(data_dicts, transform=valid_transforms)

    # Set dataloader for Trainset
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=5
    )

    return train_loader, data_dicts

dataloader, data_dict = get_dataloaders_labeled(
    root="./",
    mapping_file="./train_tools/data_utils/mapping_labeled_dic_sim.json",
    batch_size=1,
)

for i, batch in enumerate(dataloader):
    if "label" in batch:
        lbl = batch["label"].squeeze().cpu().numpy()

        if np.all(lbl == 0):
            print(f"[Zero Mask] Sample {i} has an all-zero mask!")