from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.data import Dataset
from pathlib import Path
import pickle
import numpy as np


from torch.utils.data import WeightedRandomSampler, DistributedSampler
from collections import defaultdict
import json

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


import torch
import tifffile
import os

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
    def __init__(self, data, transform=None, crop_roi=True):
        super().__init__(data, transform)
        self.data = data
        self.transform = transform
        self.crop_roi = crop_roi

    def __getitem__(self, index):
        data = dict(self.data[index])  # Make a copy

        # --- Handle paths ---
        cellcenter_path = data.get("cellcenter", None)
        if cellcenter_path is not None:
            cellcenter_path = Path(cellcenter_path)
            if not cellcenter_path.exists():
                print(f"[Warning] cellcenter file not found at index {index}: {cellcenter_path}")
            else:
                data["cellcenter"] = str(cellcenter_path)
        else:
            data.pop("cellcenter", None)

        flow_path = data.get("flow", None)
        if flow_path is not None:
            flow_path = Path(flow_path)
            if not flow_path.exists():
                print(f"[Warning] flow file not found at index {index}: {flow_path}")
                data.pop("flow", None)
            else:
                flow_np = tifffile.imread(flow_path)
                flow_tensor = torch.from_numpy(flow_np).float()
                data["flow"] = flow_tensor
        else:
            data.pop("flow", None)

        # --- Apply transform if any ---
        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.data)
    
    
    def _crop_to_ROI(self, images, labels, flows=None, center_masks=None):
        """
        Crop each image in the batch to the ROI of its label OR keep full image with probability full_prob.
        Always pads to nearest multiple of 32.
        
        images, labels, center_masks shape: [B, C, H, W]
        flows shape (if given): [B, H, W, C]
        """
        cropped_images, cropped_labels = [], []
        cropped_center_masks, cropped_flows = [], []

        for b in range(images.shape[0]):
            label = labels[b, 0]  # [H, W]
            nonzero = (label > 0).nonzero(as_tuple=False)

            # case 1: empty label -> keep full image
            if nonzero.shape[0] == 0:
                cropped_images.append(images[b])
                cropped_labels.append(labels[b])
                if center_masks is not None:
                    cropped_center_masks.append(center_masks[b])
                if flows is not None:
                    cropped_flows.append(flows[b])
                continue

            # # case 2: non-empty, maybe keep full image
            # if random.random() < full_prob:
            #     cropped_images.append(images[b])
            #     cropped_labels.append(labels[b])
            #     if center_masks is not None:
            #         cropped_center_masks.append(center_masks[b])
            #     if flows is not None:
            #         cropped_flows.append(flows[b])
            #     continue

            # case 3: ROI crop
            y_min, y_max = nonzero[:, 0].min().item(), nonzero[:, 0].max().item()
            x_min, x_max = nonzero[:, 1].min().item(), nonzero[:, 1].max().item()

            buffer = 20
            H, W = label.shape
            y_start = max(y_min - buffer, 0)
            y_end   = min(y_max + buffer, H)
            x_start = max(x_min - buffer, 0)
            x_end   = min(x_max + buffer, W)

            cropped_images.append(images[b, :, y_start:y_end, x_start:x_end])
            cropped_labels.append(labels[b, :, y_start:y_end, x_start:x_end])
            if center_masks is not None:
                cropped_center_masks.append(center_masks[b, :, y_start:y_end, x_start:x_end])
            if flows is not None:
                cropped_flows.append(flows[b, y_start:y_end, x_start:x_end, :])  # [H,W,C]

        # --- Compute max dims ---
        all_heights = [img.shape[1] for img in cropped_images]
        all_widths  = [img.shape[2] for img in cropped_images]

        if flows is not None:
            all_heights += [flow.shape[0] for flow in cropped_flows]  # [H,W,C]
            all_widths  += [flow.shape[1] for flow in cropped_flows]

        max_h, max_w = max(all_heights), max(all_widths)

        # Round up to nearest multiple of 32
        pad_h = ((max_h + 31) // 32) * 32
        pad_w = ((max_w + 31) // 32) * 32

        # --- Pad helper ---
        def pad_tensor(tensor, is_channels_last=False):
            if is_channels_last:
                # tensor shape: [H, W, C]
                h, w, c = tensor.shape
                pad_top = (pad_h - h) // 2
                pad_bottom = pad_h - h - pad_top
                pad_left = (pad_w - w) // 2
                pad_right = pad_w - w - pad_left
                padded = torch.nn.functional.pad(
                    tensor.permute(2, 0, 1),
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant', value=0
                )
                return padded.permute(1, 2, 0)  # back to [H, W, C]
            else:
                # tensor shape: [C, H, W]
                c, h, w = tensor.shape
                pad_top = (pad_h - h) // 2
                pad_bottom = pad_h - h - pad_top
                pad_left = (pad_w - w) // 2
                pad_right = pad_w - w - pad_left
                return torch.nn.functional.pad(
                    tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant', value=0
                )

        # --- Pad everything ---
        padded_images = [pad_tensor(img, is_channels_last=False) for img in cropped_images]
        padded_labels = [pad_tensor(lbl, is_channels_last=False) for lbl in cropped_labels]
        padded_center_masks = [pad_tensor(center, is_channels_last=False) for center in cropped_center_masks] if center_masks is not None else None
        padded_flows = [pad_tensor(flow, is_channels_last=True) for flow in cropped_flows] if flows is not None else None

        # --- Stack ---
        images = torch.stack(padded_images)
        labels = torch.stack(padded_labels)
        center_masks = torch.stack(padded_center_masks) if center_masks is not None else None
        flows = torch.stack(padded_flows) if flows is not None else None

        return images, labels, flows
        
        
__all__ = [
    "get_dataloaders_labeled",
    "get_dataloaders_public",
    "get_dataloaders_unlabeled",
]
def debug_dataset(dataset, num_samples=10):
    for i in range(num_samples):
        sample = dataset[i]
        print(f"--- Sample {i} ---")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {tuple(v.shape)} {v.dtype}")
            else:
                print(f"{k}: {type(v)}")

def collate_crop(batch):
    """
    Collate function for DataLoader that crops to ROI and pads all samples to the same size.
    """
    imgs = [b["img"] for b in batch]       # list of [C,H,W]
    labels = [b["label"] for b in batch]   # list of [1,H,W]

    flows = [b["flow"] for b in batch] if "flow" in batch[0] else None
    centers = [b["cellcenter"] for b in batch] if "cellcenter" in batch[0] else None

    # --- Add batch dimension for _crop_to_ROI ---
    imgs = [img.unsqueeze(0) for img in imgs]
    labels = [lbl.unsqueeze(0) for lbl in labels]
    flows = [flow.unsqueeze(0) for flow in flows] if flows is not None else None
    centers = [center.unsqueeze(0) for center in centers] if centers is not None else None

    # --- Stack lists into single tensors for _crop_to_ROI ---
    imgs_tensor = torch.cat(imgs, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    flows_tensor = torch.cat(flows, dim=0) if flows is not None else None
    centers_tensor = torch.cat(centers, dim=0) if centers is not None else None

    # --- Crop & pad ---
    cropper = CustomMediarDataset([])
    imgs_out, labels_out, flows_out = cropper._crop_to_ROI(
        imgs_tensor, labels_tensor, flows=flows_tensor, center_masks=centers_tensor
    )

    out = {"img": imgs_out, "label": labels_out}
    if flows_out is not None:
        out["flow"] = flows_out
    if centers_tensor is not None:
        out["cellcenter"] = centers_tensor  # optionally pad/stack if needed

    return out



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
    amplified=False,
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

    if amplified:
        with open(DATA_LABEL_DICT_PICKLE_FILE, "rb") as f:
            data_label_dict = pickle.load(f)

        data_point_dict = {}

        for label, data_lst in data_label_dict.items():
            data_point_dict[label] = []

            for d_idx in data_lst:
                try:
                    data_point_dict[label].append(data_dicts[d_idx])
                except:
                    print(label, d_idx)

        data_dicts = []

        for label, data_points in data_point_dict.items():
            len_data_points = len(data_points)

            if len_data_points >= 50:
                data_dicts += data_points
            else:
                for i in range(50):
                    data_dicts.append(data_points[i % len_data_points])

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
