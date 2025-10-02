import numpy as np
import torch
import tifffile as tif
import skimage.io as io
from typing import Optional, Sequence, Union
from monai.config import DtypeLike, PathLike, KeysCollection
from monai.utils import ensure_tuple
from monai.data.utils import is_supported_format, optional_import, ensure_tuple_rep
from monai.data.image_reader import ImageReader, NumpyReader
from monai.transforms import LoadImage, LoadImaged
from monai.utils.enums import PostFix
from monai.data.meta_tensor import MetaTensor
from pathlib import Path



DEFAULT_POST_FIX = PostFix.meta()
itk, has_itk = optional_import("itk", allow_namespace_pkg=True)

__all__ = [
    "CustomLoadImaged",
    "CustomLoadImageD",
    "CustomLoadImageDict",
    "CustomLoadImage",
]


# class CustomLoadImage(LoadImage):
#     """
#     Load image file or files from provided path based on reader.
#     If reader is not specified, this class automatically chooses readers
#     based on the supported suffixes and in the following order:

#         - User-specified reader at runtime when calling this loader.
#         - User-specified reader in the constructor of `LoadImage`.
#         - Readers from the last to the first in the registered list.
#         - Current default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
#           (npz, npy -> NumpyReader), (nrrd -> NrrdReader), (DICOM file -> ITKReader).

#     [!Caution] This overriding replaces the original ITK with Custom UnifiedITKReader.
#     """

#     def __init__(
#         self,
#         reader=None,
#         image_only: bool = False,
#         dtype: DtypeLike = np.float32,
#         ensure_channel_first: bool = False,
#         *args,
#         **kwargs,
#     ) -> None:
#         super(CustomLoadImage, self).__init__(
#             reader, image_only, dtype, ensure_channel_first, *args, **kwargs
#         )

#         # Adding TIFFReader. Although ITK Reader supports ".tiff" files, sometimes fails to load
#         self.readers = []
#         self.register(UnifiedITKReader(*args, **kwargs))


class CustomLoadImage(LoadImage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.readers = self.readers + [UnifiedITKReader(*args, **kwargs)] #idk why suddently dicom reader grabs my image but appearently images.LoadImage.__call_ MONAI loops over self.readers[::-1] (reversed order) so my UnifiedITKRead should be appended first.

    def __call__(self, filename, reader=None):
        img = super().__call__(filename, reader)
        if isinstance(img, MetaTensor):
            img.meta["filename_or_obj"] = filename
        return img
    
class CustomLoadImaged(LoadImaged):
    def __init__(self, keys, image_only=True, allow_missing_keys=False, *args, **kwargs):
        super().__init__(keys, image_only=image_only, allow_missing_keys=allow_missing_keys, *args, **kwargs)
        self._loader = CustomLoadImage(image_only=image_only)
        self._loader.readers = [UnifiedITKReader(*args, **kwargs)] + self._loader.readers

    def __call__(self, data):
        for key in self.keys:
            if key not in data:
                continue

            val = data[key]

            # Skip if already tensor or numpy array (already loaded)
            if isinstance(val, (torch.Tensor, np.ndarray, MetaTensor)):
                continue

            # If it's a path (string/Path), then load
            if isinstance(val, (str, Path)):
                try:
                    loaded = self._loader(val)
                    if self._loader.image_only and isinstance(loaded, dict):
                        loaded = loaded["image"]
                    data[key] = loaded
                except Exception as e:
                    if self.allow_missing_keys:
                        continue
                    raise e

        return data
    
class UnifiedITKReader(NumpyReader):
    """
    Unified Reader to read ".tif" and ".tiff files".
    As the tifffile reads the images as numpy arrays, it inherits from the NumpyReader.
    """

    def __init__(
        self, channel_dim: Optional[int] = None, **kwargs,
    ):
        super(UnifiedITKReader, self).__init__(channel_dim=channel_dim, **kwargs)
        self.kwargs = kwargs
        self.channel_dim = channel_dim

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """Verify whether the file format is supported by TIFF Reader."""

        suffixes: Sequence[str] = ["tif", "tiff", "png", "jpg", "bmp", "jpeg",]
        return has_itk or is_supported_format(filename, suffixes)
    
    def move_channel_last(self, axis, obj):
        """Puts the channel axis to last position (works for torch.Tensor or np.ndarray)."""
        order = [j for j in range(obj.ndim) if j != axis] + [axis]
        
        if isinstance(obj, torch.Tensor):
            return obj.permute(*order)
        elif isinstance(obj, np.ndarray):
            return np.transpose(obj, order)
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
        """Read Images from the file."""
        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)

        for name in filenames:
            name = f"{name}"

            if name.endswith(".tif") or name.endswith(".tiff"):
                _obj = tif.imread(name)
            else:
                try:
                    _obj = itk.imread(name, **kwargs_)
                    _obj = itk.array_view_from_image(_obj, keep_axes=False)
                except:
                    _obj = io.imread(name)
            meta = {
                "spatial_shape": _obj.shape,  # shape before EnsureChannelFirst
                "filename_or_obj": name,
            }
            if len(_obj.shape) == 2:
                meta["dimensionality"] = 2
                _obj = np.repeat(np.expand_dims(_obj, axis=-1), 3, axis=-1) # (H, W, 3)
            elif len(_obj.shape) == 3:
                if _obj.shape[0] > 3 and _obj.shape[-1] > 3:  # heuristically a (Z, H, W), add channel dimension
                    meta["dimensionality"] = 3
                    _obj = np.expand_dims(_obj, axis=-1)  # (Z, H, W, 1)
                    #_obj = np.repeat(np.expand_dims(_obj, axis=-1), 3, axis=-1)  # (Z, H, W, 3)
                
                else: 
                    for p in range(3):
                        if _obj.shape[p] == 1:
                            meta["dimensionality"] = 2
                            _obj = np.repeat(_obj, 3, axis=p)  
                            _obj = self.move_channel_last(p, _obj)
                        elif _obj.shape[p] == 3:
                            meta["dimensionality"] = 2
                            _obj = self.move_channel_last(p, _obj)

                # else, leave it alone if already fine
            elif len(_obj.shape) == 4:
                if _obj.shape[0] > 3 and _obj.shape[-1] > 3:  # heuristically a (Z, H, W)
                    meta["dimensionality"] = 3
                    _obj = _obj[..., :3]  # (Z, H, W, 3)
                else:
                    for p in range(4):
                        if _obj.shape[p] == 1:
                            meta["dimensionality"] = 3
                            _obj = np.repeat(_obj, 3, axis=p)  
                            _obj = self.move_channel_last(p, _obj)
                        if _obj.shape[p] == 3:
                            meta["dimensionality"] = 3
                            _obj = self.move_channel_last(p, _obj)
            img_.append(_obj)

        return img_ if len(filenames) > 1 else img_[0]


CustomLoadImageD = CustomLoadImageDict = CustomLoadImaged
