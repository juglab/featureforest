from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
import pims
import torch
from tifffile import natural_sorted
from torch.utils.data import IterableDataset

from featureforest.utils.data import (
    get_model_ready_image,
    patchify,
)


class FFImageDataset(IterableDataset):
    """
    Iterable dataset for large images or image stacks.
    This dataset can handle large TIFF files or directories of images,
    and it can yield patches of images or the whole image depending on the configuration.
    """

    def __init__(
        self,
        image_array: np.ndarray | None = None,
        stack_file: str | Path | None = None,
        img_dir: str | Path | None = None,
        no_patching: bool = False,
        patch_size: int = 512,
        overlap: int = 128,
    ) -> None:
        super().__init__()
        if image_array is not None and (stack_file is not None or img_dir is not None):
            raise ValueError(
                "Please provide either an image array or a stack file or an image directory, not both."
            )
        if image_array is None and stack_file is None and img_dir is None:
            raise ValueError(
                "Please provide either a large TIFF file or an image directory."
            )

        self.no_patching = no_patching
        self.patch_size = patch_size
        self.overlap = overlap
        self.image_files = []
        self.image_source: Optional[pims.ImageSequence | np.ndarray] = None

        if image_array is not None:
            # image is already loaded as a numpy array
            self.image_source = image_array
            # add slice dimension if not present
            if self.image_source.ndim == 2:
                self.image_source = self.image_source[np.newaxis, ...]
        elif stack_file is not None:
            # can be a large stack, using pims for lazy loading
            self.image_source = pims.open(str(stack_file))
        elif img_dir is not None:
            # load images from a directory
            img_dir = Path(img_dir)
            if not img_dir.is_dir():
                raise ValueError(f"The provided path {img_dir} is not a directory.")
            self.image_files = (
                list(img_dir.glob("*.tiff"))
                + list(img_dir.glob("*.tif"))
                + list(img_dir.glob("*.png"))
                + list(img_dir.glob("*.jpg"))
            )
            if not self.image_files:
                raise ValueError(f"No image files found in the directory {img_dir}.")
            self.image_files = self._natural_sort(self.image_files)
            self.image_source = pims.ImageSequence(map(str, self.image_files))

    @property
    def num_images(self) -> int:
        """Return the number of images in the dataset."""
        if self.image_source is None:
            return 0
        return len(self.image_source)

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return the shape of the images in the dataset."""
        if self.image_source is None:
            raise ValueError("No image source is available. Please check the input data.")
        if isinstance(self.image_source, np.ndarray):
            return self.image_source.shape[1:]
        return self.image_source.frame_shape

    def __iter__(self):
        if self.image_source is None:
            raise ValueError("No image source is available. Please check the input data.")

        for img_idx, img_slice in enumerate(self.image_source):
            img_tensor = get_model_ready_image(img_slice)
            if self.no_patching:
                # return the whole image as a tensor
                yield img_tensor, torch.tensor([img_idx, 0])
            else:
                # divide the image into patches and yield them
                patches = patchify(img_tensor, self.patch_size, self.overlap)
                for p_idx, patch in enumerate(patches):
                    yield patch, torch.tensor([img_idx, p_idx])

    def _natural_sort(self, files: Iterable[Path]) -> list[Path]:
        """Sort files in a natural order."""
        return sorted(files, key=lambda x: natural_sorted(str(x.name)))
