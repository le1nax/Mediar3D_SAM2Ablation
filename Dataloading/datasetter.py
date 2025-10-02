from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.data import Dataset
from pathlib import Path
import pickle
import numpy as np

from torch.utils.data import WeightedRandomSampler, DistributedSampler
from collections import defaultdict
import json, os, torch
import tifffile as tif


from train_tools.data_utils.transforms import (
    train_transforms,
    masked_train_transforms,
    public_transforms,
    valid_transforms,
    tuning_transforms,
    unlabeled_transforms,
)
from train_tools.data_utils.utils import split_train_valid, path_decoder, add_flows

DATA_LABEL_DICT_PICKLE_FILE = "./train_tools/data_utils/custom/modalities.pkl"



# class CustomMediarDataset(Dataset):
#     def __init__(self, data, transform=None):
#         super().__init__(data, transform)

#     def __getitem__(self, index):
#         data = dict(self.data[index])  # Make a copy

#         # Check if 'cellcenter' is present
#         cellcenter_path = data.get("cellcenter", None)
#         if cellcenter_path is not None:
#             cellcenter_path = Path(cellcenter_path)
#             if not cellcenter_path.exists():
#                 print(f"[Warning] cellcenter file not found at index {index}: {cellcenter_path}")
#             else:
#                 data["cellcenter"] = str(cellcenter_path)  # Normalize path if needed
#         else:
#             # Remove the key if it's None to avoid transform issues
#             data.pop("cellcenter", None)

#         if self.transform:
#             data = self.transform(data)

#         return data
    
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
                flow_np = tif.imread(flow_path)
                flow_tensor = torch.from_numpy(flow_np).float()
                data["flow"] = flow_tensor
        else:
            data.pop("flow", None)
            
        if self.transform:
            data = self.transform(data)

        return data
        
__all__ = [
    "get_dataloaders_labeled",
    "get_dataloaders_public",
    "get_dataloaders_unlabeled",
]


# ---------------------------
# Grouping helpers
# ---------------------------
def load_and_group_by_dataset(data_dicts):
    grouped = defaultdict(list)
    for sample in data_dicts:
        dataset_name = sample["img"].split("/")[-2]
        grouped[dataset_name].append(sample)
    return grouped


def make_weighted_sampler(dataset, grouped, custom_ratios):
    total_ratio = sum(custom_ratios.values())
    ratios = {k: v / total_ratio for k, v in custom_ratios.items()}
    dataset_sizes = {k: len(v) for k, v in grouped.items()}

    weights = torch.zeros(len(dataset))
    for idx, sample in enumerate(dataset.data):
        dataset_name = sample["img"].split("/")[-2]
        weights[idx] = ratios[dataset_name] / dataset_sizes[dataset_name]

    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


# ---------------------------
# Integrated get_dataloaders_labeled
# ---------------------------
def get_dataloaders_labeled_sampled(
    root,
    mapping_file,
    tuning_mapping_file,
    join_mapping_file=None,
    valid_portion=0.0,
    batch_size=8,
    sampling_ratios=None,      # <-- now a dict (dataset_name: ratio)
    relabel=False,
    precompute_flows=False,
    incomplete_annotations=False,
    distributed=False,
    rank=0,
    world_size=1,
):
    # --- Load mapping files ---
    data_dicts = path_decoder(root, mapping_file)
    data_dicts = add_flows(data_dicts, device="cuda", overwrite=False, precompute_flows=precompute_flows)
    tuning_dicts = path_decoder(root, tuning_mapping_file, no_label=True)

    if incomplete_annotations:
        data_transforms = masked_train_transforms
    else:
        data_transforms = train_transforms

    if join_mapping_file is not None:
        data_dicts += path_decoder(root, join_mapping_file)
        data_transforms = public_transforms

    # --- Split train/valid ---
    train_dicts, valid_dicts = split_train_valid(
        data_dicts, valid_portion=valid_portion
    )

    # --- Create Datasets ---
    trainset = CustomMediarDataset(train_dicts, transform=data_transforms)
    validset = CustomMediarDataset(valid_dicts, transform=valid_transforms)
    tuningset = Dataset(tuning_dicts, transform=tuning_transforms)

    # --- Sampler logic ---
    train_sampler, valid_sampler = None, None

    if isinstance(sampling_ratios, dict) and len(sampling_ratios) > 0:
        grouped = load_and_group_by_dataset(train_dicts)
        train_sampler = make_weighted_sampler(trainset, grouped, sampling_ratios)
    elif distributed:  # distributed
        train_sampler = DistributedSampler(
            trainset, num_replicas=world_size, rank=rank, shuffle=True
        )
        if len(validset) > 0:
            valid_sampler = DistributedSampler(validset, num_replicas=world_size, rank=rank, shuffle=False)

    # --- DataLoaders ---
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # shuffle if no sampler
        sampler=train_sampler,
        num_workers=5,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        validset,
        batch_size=1,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=2,
        pin_memory=True,
    )

    tuning_loader = DataLoader(
        tuningset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    return {
        "train": train_loader,
        "valid": valid_loader,
        "tuning": tuning_loader,
    }


def get_dataloaders_labeled(
    root,
    mapping_file,
    tuning_mapping_file,
    join_mapping_file=None,
    valid_portion=0.0,
    batch_size=8,
    sampling_ratios=False,
    relabel=False,
    precompute_flows=False,
    incomplete_annotations=False,
    distributed=False,   # <-- NEW ARG
    rank=0,              # <-- NEW ARG
    world_size=1,        # <-- NEW ARG
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
    data_dicts = add_flows(data_dicts, device="cuda", overwrite=False, precompute_flows=precompute_flows)
    tuning_dicts = path_decoder(root, tuning_mapping_file, no_label=True)


    if incomplete_annotations:
        data_transforms = masked_train_transforms
    else:
        data_transforms = train_transforms

    if join_mapping_file is not None:
        data_dicts += path_decoder(root, join_mapping_file)
        data_transforms = public_transforms

    if relabel:
        for elem in data_dicts:
            cell_idx = int(elem["label"].split("_label.tiff")[0].split("_")[-1])
            if cell_idx in range(340, 499):
                new_label = elem["label"].replace(
                    "/data/CellSeg/Official/Train_Labeled/labels/",
                    "/CellSeg/pretrained_train_ext/",
                )
                elem["label"] = new_label

    # Split datasets as Train/Valid
    train_dicts, valid_dicts = split_train_valid(
        data_dicts, valid_portion=valid_portion
    )

     # Datasets
    trainset = CustomMediarDataset(train_dicts, transform=data_transforms)
    validset = CustomMediarDataset(valid_dicts, transform=valid_transforms)
    tuningset = Dataset(tuning_dicts, transform=tuning_transforms)

    if distributed:
        train_sampler = DistributedSampler(
            trainset, num_replicas=world_size, rank=rank, shuffle=True
        )
        valid_sampler = DistributedSampler(
            validset, num_replicas=world_size, rank=rank, shuffle=False
        ) if len(validset) > 0 else None
    else:
        train_sampler, valid_sampler = None, None

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=5,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        validset,
        batch_size=1,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=2,
        pin_memory=True,
    )

    tuning_loader = DataLoader(
        tuningset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "tuning": tuning_loader,
    }
    return dataloaders



def get_dataloaders_public(
    root, mapping_file, valid_portion=0.0, batch_size=8,
):
    """Set DataLoaders for labeled datasets.

    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
        shuffle (bool, optional): shuffles dataloader. Defaults to True.

    Returns:
        dict: dictionary of data loaders.
    """

    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file)

    # Split datasets as Train/Valid
    train_dicts, _ = split_train_valid(data_dicts, valid_portion=valid_portion)

    trainset = Dataset(train_dicts, transform=public_transforms)
    # Set dataloader for Trainset
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=5
    )

    # Form dataloaders as dictionary
    dataloaders = {
        "public": train_loader,
    }

    return dataloaders


def get_dataloaders_unlabeled(
    root, mapping_file, batch_size=8, shuffle=True, num_workers=5,
):
    """Set dataloaders for unlabeled dataset."""
    # Get list of data dictionaries from decoded paths
    unlabeled_dicts = path_decoder(root, mapping_file, no_label=True, unlabeled=True)

    # Obtain datasets with transforms
    unlabeled_dicts, _ = split_train_valid(unlabeled_dicts, valid_portion=0)
    unlabeled_set = Dataset(unlabeled_dicts, transform=unlabeled_transforms)

    # Set dataloader for Unlabeled dataset
    unlabeled_loader = DataLoader(
        unlabeled_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    dataloaders = {
        "unlabeled": unlabeled_loader,
    }

    return dataloaders


def get_dataloaders_unlabeled_psuedo(
    root, mapping_file, batch_size=8, shuffle=True, num_workers=5,
):

    # Get list of data dictionaries from decoded paths
    unlabeled_psuedo_dicts = path_decoder(
        root, mapping_file, no_label=False, unlabeled=True
    )

    # Obtain datasets with transforms
    unlabeled_psuedo_dicts, _ = split_train_valid(
        unlabeled_psuedo_dicts, valid_portion=0
    )
    unlabeled_psuedo_set = Dataset(unlabeled_psuedo_dicts, transform=train_transforms)

    # Set dataloader for Unlabeled dataset
    unlabeled_psuedo_loader = DataLoader(
        unlabeled_psuedo_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    dataloaders = {"unlabeled": unlabeled_psuedo_loader}

    return dataloaders
