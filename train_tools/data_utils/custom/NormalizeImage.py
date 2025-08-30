import numpy as np
import torch
from skimage import exposure
from monai.config import KeysCollection

from monai.transforms.transform import Transform
from monai.transforms.compose import MapTransform

from typing import Dict, Hashable, Mapping


__all__ = [
    "CustomNormalizeImage",
    "CustomNormalizeImageD",
    "CustomNormalizeImageDict",
    "CustomNormalizeImaged",
]

def safe_percentiles(img, lower=1, upper=99, sample_size=500_000):
    # Ensure numpy float32
    if torch.is_tensor(img):
        img = img.cpu().numpy().astype(np.float32, copy=False)

    flat = img.ravel()
    n = flat.size

    if n > sample_size:
        idx = np.random.randint(0, n, size=sample_size, dtype=np.int64)
        sample = flat[idx]
    else:
        sample = flat

    # Ignore zeros
    sample = sample[sample > 0]
    if sample.size == 0:
        return 0, 1

    return np.percentile(sample, [lower, upper])

def rescale_intensity_np(img, low, high, out_dtype=np.uint8):
    img = img.astype(np.float32, copy=False)  # avoid float64
    if high > low:
        img = (img - low) / (high - low + 1e-8)
        img = np.clip(img, 0, 1)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    img = (img * np.iinfo(out_dtype).max).astype(out_dtype, copy=False)
    return img

class CustomNormalizeImage(Transform):
    """Memory-safe percentile normalization for large images."""

    def __init__(self, percentiles=[0, 99.5], channel_wise=False, sample_size=500_000):
        """
        percentiles: [lower, upper]
        channel_wise: whether to normalize each channel independently
        sample_size: number of voxels to sample for percentile computation
        """
        self.lower, self.upper = percentiles
        self.channel_wise = channel_wise
        self.sample_size = sample_size

    @staticmethod
    def safe_percentiles(img, lower, upper, sample_size=500_000):
        """Compute approximate percentiles using a random sample of non-zero voxels."""
        flat = img.ravel()
        n = flat.size
        if n > sample_size:
            idx = np.random.randint(0, n, size=sample_size, dtype=np.int64)
            sample = flat[idx]
        else:
            sample = flat
        sample = sample[sample > 0]
        if sample.size == 0:
            return 0.0, 1.0
        return np.percentile(sample, [lower, upper])

    @staticmethod
    def inplace_rescale(img, low, high, out_dtype=np.uint8):
        """Normalize in-place to [0, 255] (or other dtype)."""
        if high > low:
            img -= low
            img /= (high - low + 1e-8)
            np.clip(img, 0.0, 1.0, out=img)
        else:
            img.fill(0.0)
        img *= np.iinfo(out_dtype).max
        img[:] = img.astype(out_dtype, copy=False)

    def _normalize_volume(self, img):
        """Normalize entire volume at once (per-channel if requested)."""
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        if np.all(img == 0):
            return np.zeros_like(img, dtype=np.uint8)

        if self.channel_wise and img.ndim == 3:  # HWC
            for c in range(img.shape[-1]):
                channel = img[..., c]
                if np.any(channel > 0):
                    low, high = self.safe_percentiles(channel, self.lower, self.upper, self.sample_size)
                    self.inplace_rescale(channel, low, high)
            return img
        else:
            low, high = self.safe_percentiles(img, self.lower, self.upper, self.sample_size)
            self.inplace_rescale(img, low, high)
            return img

    def __call__(self, img):
        return self._normalize_volume(img)


class CustomNormalizeImaged(MapTransform):
    """Dictionary-based wrapper of CustomNormalizeImage"""

    def __init__(
        self,
        keys: KeysCollection,
        percentiles=[1, 99],
        channel_wise: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.normalizer = CustomNormalizeImage(percentiles, channel_wise)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.normalizer(d[key])
        return d


CustomNormalizeImageD = CustomNormalizeImageDict = CustomNormalizeImaged
