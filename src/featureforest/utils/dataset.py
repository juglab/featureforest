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
    is_stacked,
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
        images: str | Path | np.ndarray,
        no_patching: bool = False,
        patch_size: int = 512,
        overlap: int = 128,
    ) -> None:
        super().__init__()
        self.no_patching = no_patching
        self.patch_size = patch_size
        self.overlap = overlap
        self.image_files = []
        self.image_source: Optional[pims.ImageSequence | np.ndarray] = None

        if isinstance(images, np.ndarray):
            # images are already loaded into a numpy array
            self.image_source = images
            # add slice dimension if not present
            if not is_stacked(self.image_source):
                self.image_source = self.image_source[np.newaxis, ...]

        elif isinstance(images, str | Path):
            images = Path(images)
            if images.is_file():
                # can be a large stack, using pims for lazy loading
                self.image_source = pims.open(str(images))

            elif images.is_dir():
                self.image_files = (
                    list(images.glob("*.tiff"))
                    + list(images.glob("*.tif"))
                    + list(images.glob("*.png"))
                    + list(images.glob("*.jpg"))
                )
                if not self.image_files:
                    raise ValueError(f"No image files found in the directory {images}.")
                self.image_files = self._natural_sort(self.image_files)
                self.image_source = pims.ImageSequence(map(str, self.image_files))
        else:
            raise ValueError(
                f"images should be a numpy array or a directory or an image stack!"
                f"\nGot {type(images)}"
            )

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
