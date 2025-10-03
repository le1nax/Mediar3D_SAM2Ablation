from .custom import *
from .custom.LoadImage import SimpleLoadImaged

from monai.transforms import *
from pathlib import Path

import sys
import gc
import psutil
import tracemalloc

__all__ = [
    "SimpleLoadImaged"
    "train_transforms",
    "public_transforms",
    "valid_transforms",
    "tuning_transforms",
    "unlabeled_transforms",
]

def log_mem(tag):
    process = psutil.Process()
    mem = process.memory_info().rss / 1e9
    current, peak = tracemalloc.get_traced_memory()
    print(f"[{tag}] RAM: {mem:.2f} GB | Python objs: {current/1e6:.1f} MB (peak {peak/1e6:.1f} MB)")
    gc.collect()

class DebugCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            log_mem(t.__class__.__name__)
        return data

# replace Compose with DebugCompose
debug_train_transforms = DebugCompose(
[
        CustomLoadImaged(keys=["img", "label", "cellcenter"], image_only=True, allow_missing_keys=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label", "cellcenter", "flow"], channel_dim=-1, allow_missing_keys=True),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),

        RandZoomd(
            keys=["img", "label", "cellcenter", "flow"],
            prob=0.5,
            min_zoom=0.25,
            max_zoom=1.5,
            mode=["area", "nearest", "nearest", "area"],
            keep_size=False,
            allow_missing_keys=True,
        ),
        SpatialPadd(keys=["img", "label", "cellcenter", "flow"], spatial_size=512, allow_missing_keys=True),
        RandSpatialCropd(keys=["img", "label", "cellcenter", "flow"], roi_size=512, random_size=False, allow_missing_keys=True),
        RandAxisFlipd(keys=["img", "label", "cellcenter", "flow"], prob=0.5, allow_missing_keys=True),
        RandRotate90d(keys=["img", "label", "cellcenter", "flow"], prob=0.5, spatial_axes=[0, 1], allow_missing_keys=True),

        IntensityDiversification(keys=["img", "label"], allow_missing_keys=True),
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        RandGaussianSharpend(keys=["img"], prob=0.25),
       # EnsureTyped(keys=["img", "label", "cellcenter", "flow"], allow_missing_keys=True),
    ]
)

# train_transforms = Compose(
#     [
#         CustomLoadImaged(keys=["img", "label", "cellcenter"], image_only=True, allow_missing_keys=True),
#         CustomNormalizeImaged(
#             keys=["img"],
#             allow_missing_keys=True,
#             channel_wise=False,
#             percentiles=[0.0, 99.5],
#         ),
#         EnsureChannelFirstd(keys=["img", "label", "cellcenter", "flow"], channel_dim=-1, allow_missing_keys=True),
#         RemoveRepeatedChanneld(keys=["label"], repeats=3),
#         ScaleIntensityd(keys=["img"], allow_missing_keys=True),

#         RandZoomd(
#             keys=["img", "label", "cellcenter", "flow"],
#             prob=0.5,
#             min_zoom=0.25,
#             max_zoom=1.5,
#             mode=["area", "nearest", "nearest", "area"],
#             keep_size=False,
#             allow_missing_keys=True,
#         ),
#         SpatialPadd(keys=["img", "label", "cellcenter", "flow"], spatial_size=512, allow_missing_keys=True),
#         RandSpatialCropd(keys=["img", "label", "cellcenter", "flow"], roi_size=512, random_size=False, allow_missing_keys=True),
#         RandAxisFlipd(keys=["img", "label", "cellcenter", "flow"], prob=0.5, allow_missing_keys=True),
#         RandRotate90d(keys=["img", "label", "cellcenter", "flow"], prob=0.5, spatial_axes=[0, 1], allow_missing_keys=True),

#         IntensityDiversification(keys=["img", "label"], allow_missing_keys=True),
#         RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
#         RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
#         RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
#         RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
#         RandGaussianSharpend(keys=["img"], prob=0.25),
#        EnsureTyped(keys=["img", "label", "cellcenter", "flow"], allow_missing_keys=True),
#     ]
# )

train_transforms = Compose(
    [
        SimpleLoadImaged(keys=["img", "label"], image_only=True, allow_missing_keys=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label", "cellcenter", "flow"], channel_dim=-1, allow_missing_keys=True),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),

        RandZoomd(
            keys=["img", "label", "cellcenter", "flow"],
            prob=0.5,
            min_zoom=0.25,
            max_zoom=1.5,
            mode=["area", "nearest", "nearest", "area"],
            keep_size=False,
            allow_missing_keys=True,
        ),
        SpatialPadd(keys=["img", "label", "cellcenter", "flow"], spatial_size=512, allow_missing_keys=True),
        RandSpatialCropd(keys=["img", "label", "cellcenter", "flow"], roi_size=512, random_size=False, allow_missing_keys=True),
        RandAxisFlipd(keys=["img", "label", "cellcenter", "flow"], prob=0.5, allow_missing_keys=True),
        RandRotate90d(keys=["img", "label", "cellcenter", "flow"], prob=0.5, spatial_axes=[0, 1], allow_missing_keys=True),

        IntensityDiversification(keys=["img", "label"], allow_missing_keys=True),
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        RandGaussianSharpend(keys=["img"], prob=0.25),
       EnsureTyped(keys=["img", "label", "cellcenter", "flow"], allow_missing_keys=True),
        
    ]
)


# train_transforms = Compose(
#     [
#         # >>> Load and refine data --- img: (H, W, 3); label: (H, W)
#         CustomLoadImaged(keys=["img", "label"], image_only=True),
#         CustomNormalizeImaged(
#             keys=["img"],
#             allow_missing_keys=True,
#             channel_wise=False,
#             percentiles=[0.0, 99.5],
#         ),
#         EnsureChannelFirstd(keys=["img", "label"], channel_dim=-1),
#         RemoveRepeatedChanneld(keys=["label"], repeats=3),  # label: (H, W)
#         ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
#         # >>> Spatial transforms
#         RandZoomd(
#             keys=["img", "label"],
#             prob=0.5,
#             min_zoom=0.25,
#             max_zoom=1.5,
#             mode=["area", "nearest"],
#             keep_size=False,
#         ),
#         SpatialPadd(keys=["img", "label"], spatial_size=512),
#         RandSpatialCropd(keys=["img", "label"], roi_size=512, random_size=False),
#         RandAxisFlipd(keys=["img", "label"], prob=0.5),
#         RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
#         IntensityDiversification(keys=["img", "label"], allow_missing_keys=True),
#         # # >>> Intensity transforms
#         RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
#         RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
#         RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
#         RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
#         RandGaussianSharpend(keys=["img"], prob=0.25),
#         EnsureTyped(keys=["img", "label"]),
#     ]
# )





masked_train_transforms = Compose(
    [
        CustomLoadImaged(keys=["img", "label", "cellcenter"], image_only=True, allow_missing_keys=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label", "cellcenter", "flow"], channel_dim=-1, allow_missing_keys=True),
        RemoveRepeatedChanneld(keys=["label"], repeats=3, allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),

        # RandZoomd(
        #     keys=["img", "label", "cellcenter", "flow"],
        #     prob=0.5,
        #     min_zoom=(0.25, 0.25),
        #     max_zoom=(1.5, 1.5),
        #     mode=["area", "nearest", "nearest", "area"],
        #     keep_size=False,
        #     allow_missing_keys=True,
        # ),
        #RandSpatialCropd(keys=["img", "label", "cellcenter", "flow"], roi_size=512, random_size=False, allow_missing_keys=True),
        RandAxisFlipd(keys=["img", "label", "cellcenter", "flow"], prob=0.5, allow_missing_keys=True),
        RandRotate90d(keys=["img", "label", "cellcenter", "flow"], prob=0.5, spatial_axes=[0, 1], allow_missing_keys=True),

        IntensityDiversification(keys=["img", "label"], allow_missing_keys=True),
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        RandGaussianSharpend(keys=["img"], prob=0.25),
        EnsureTyped(keys=["img", "label", "cellcenter", "flow"], allow_missing_keys=True),
        SpatialPadd(keys=["img", "label", "cellcenter", "flow"], spatial_size=2000, allow_missing_keys=True)
    ]
)





public_transforms = Compose(
    [
        CustomLoadImaged(keys=["img", "label"], image_only=True),
        BoundaryExclusion(keys=["label"]),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label"], channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),  # label: (H, W)
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        SpatialPadd(keys=["img", "label"], spatial_size=512),
        RandSpatialCropd(keys=["img", "label"], roi_size=512, random_size=False),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        Rotate90d(k=1, keys=["label"], spatial_axes=(0, 1)),
        Flipd(keys=["label"], spatial_axis=0),
        EnsureTyped(keys=["img", "label"]),
    ]
)


valid_transforms = Compose(
    [
        CustomLoadImaged(keys=["img", "label", "cellcenter"], allow_missing_keys=True, image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img", "label", "cellcenter", "flow"], allow_missing_keys=True, channel_dim=-1),
        RemoveRepeatedChanneld(keys=["label"], repeats=3),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        EnsureTyped(keys=["img", "label", "cellcenter", "flow"], allow_missing_keys=True),
       # SpatialPadd(keys=["img", "label", "cellcenter", "flow"], spatial_size=2000, allow_missing_keys=True),
    ]
)

tuning_transforms = Compose(
    [
        CustomLoadImaged(keys=["img"], image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img"], channel_dim=-1),
        ScaleIntensityd(keys=["img"]),
        EnsureTyped(keys=["img"]),
    ]
)

unlabeled_transforms = Compose(
    [
        # >>> Load and refine data --- img: (H, W, 3); label: (H, W)
        CustomLoadImaged(keys=["img"], image_only=True),
        CustomNormalizeImaged(
            keys=["img"],
            allow_missing_keys=True,
            channel_wise=False,
            percentiles=[0.0, 99.5],
        ),
        EnsureChannelFirstd(keys=["img"], channel_dim=-1),
        RandZoomd(
            keys=["img"],
            prob=0.5,
            min_zoom=0.25,
            max_zoom=1.25,
            mode=["area"],
            keep_size=False,
        ),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),  # Do not scale label
        # >>> Spatial transforms
        SpatialPadd(keys=["img"], spatial_size=512),
        RandSpatialCropd(keys=["img"], roi_size=512, random_size=False),
        EnsureTyped(keys=["img"]),
    ]
)


def get_pred_transforms():
    """Prediction preprocessing"""
    pred_transforms = Compose(
        [
            # >>> Load and refine data
            CustomLoadImage(image_only=True),
            CustomNormalizeImage(channel_wise=False, percentiles=[0.0, 99.5]),
            EnsureChannelFirst(channel_dim=-1),  # image: (3, H, W)
            ScaleIntensity(),
            EnsureType(data_type="tensor"),
        ]
    )

    return pred_transforms

def get_pred_transforms_3D():
    """Prediction preprocessing"""
    pred_transforms = Compose(
        [
            # >>> Load and refine data
            CustomLoadImage(image_only=True),
            CustomNormalizeImage(channel_wise=False, percentiles=[0.0, 99.5]),
            # @todo, take care of 3d #EnsureChannelFirst(channel_dim=-1),  # image: (3, H, W)
            ScaleIntensity(),
            EnsureType(data_type="tensor"),
        ]
    )

    return pred_transforms

