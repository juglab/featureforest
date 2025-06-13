from collections.abc import Generator
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import zarr
import zarr.core
import zarr.storage
from numcodecs import Zstd
from torch.utils.data import DataLoader

from featureforest.models import BaseModelAdapter
from featureforest.models.SAM import SAMAdapter
from featureforest.utils.dataset import FFImageDataset


def get_batch_size(model_adapter: BaseModelAdapter) -> int:
    """Get the batch size for the model adapter.
    The batch size is set to 8 for most models, but for SAMAdapter it is set to 2
    to avoid memory issues with large images.
    Args:
        model_adapter (BaseModelAdapter): The model adapter to get the batch size for.
    Returns:
        int: The batch size for the model adapter.
    """
    # set a low batch size
    batch_size = 8
    # for big SAM we need even lower batch size :(
    if isinstance(model_adapter, SAMAdapter):
        batch_size = 2
    return batch_size


def get_image_dataset(
    image: str | np.ndarray,
    no_patching: bool = False,
    patch_size: int = 512,
    overlap: int = 128,
) -> FFImageDataset:
    if isinstance(image, str):
        # image is a path to a large TIFF file or a directory of images
        img_path = Path(image)
        if img_path.is_dir():
            # load images from a directory
            dataset = FFImageDataset(
                img_dir=img_path,
                no_patching=no_patching,
                patch_size=patch_size,
                overlap=overlap,
            )
        else:
            # load a (large) stack
            dataset = FFImageDataset(
                stack_file=img_path,
                no_patching=no_patching,
                patch_size=patch_size,
                overlap=overlap,
            )
    elif isinstance(image, np.ndarray):
        # image is already loaded as a numpy array
        dataset = FFImageDataset(
            image_array=image,
            no_patching=no_patching,
            patch_size=patch_size,
            overlap=overlap,
        )

    return dataset


def extract_embeddings(
    image: np.ndarray | str,
    model_adapter: BaseModelAdapter,
    image_dataset: Optional[FFImageDataset] = None,
) -> Generator[tuple[np.ndarray, int, int], None, None]:
    no_patching = model_adapter.no_patching
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    batch_size = get_batch_size(model_adapter)

    # create the image dataset
    if image_dataset is None:
        image_dataset = get_image_dataset(
            image=image, no_patching=no_patching, patch_size=patch_size, overlap=overlap
        )
    # loop through the dataset and extract features
    dataloader = DataLoader(
        image_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    for img_data, indices in dataloader:
        features = model_adapter.get_features_patches(img_data.to(model_adapter.device))
        unique_slices = torch.unique(indices[:, 0]).numpy()
        for idx in unique_slices:
            img_features = features[indices[:, 0] == idx].numpy().astype(np.float16)

            yield img_features, idx, image_dataset.num_images


def extract_embeddings_to_file(
    image: np.ndarray | str, storage_path: str, model_adapter: BaseModelAdapter
) -> Generator[tuple[int, int], None, None]:
    no_patching = model_adapter.no_patching
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    # batch_size = get_batch_size(model_adapter)

    # # create the image dataset
    image_dataset = get_image_dataset(
        image=image, no_patching=no_patching, patch_size=patch_size, overlap=overlap
    )
    # create the zarr storage
    storage = zarr.storage.DirectoryStore(storage_path)
    store_root = zarr.group(store=storage, overwrite=False)
    store_root.attrs["num_slices"] = image_dataset.num_images
    store_root.attrs["img_height"] = image_dataset.image_shape[0]
    store_root.attrs["img_width"] = image_dataset.image_shape[1]
    store_root.attrs["model"] = model_adapter.name
    store_root.attrs["no_patching"] = no_patching
    store_root.attrs["patch_size"] = patch_size
    store_root.attrs["overlap"] = overlap

    for img_features, idx, total in extract_embeddings(
        image, model_adapter, image_dataset
    ):
        if store_root.get(str(idx)) is None:
            # create a group for the slice
            grp = store_root.create_group(str(idx))  # type: ignore
            z_arr = grp.create(  # type: ignore
                name="features",
                shape=img_features.shape,
                chunks=(1,) + img_features.shape[1:],
                dtype=np.float16,
                compressor=Zstd(level=3),
            )
            z_arr[:] = img_features
        else:
            # append features to the slice/image group
            grp: zarr.core.Array = store_root[str(idx)]["features"]  # type: ignore
            grp.append(img_features)

        yield idx, total

    storage.close()
