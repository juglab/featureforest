from collections.abc import Generator
from typing import Optional

import h5py
import numpy as np
import torch
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


def extract_embeddings(
    model_adapter: BaseModelAdapter,
    image: Optional[np.ndarray | str] = None,
    image_dataset: Optional[FFImageDataset] = None,
) -> Generator[tuple[np.ndarray, int, int], None, None]:
    no_patching = model_adapter.no_patching
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    batch_size = get_batch_size(model_adapter)

    # create the image dataset
    if image_dataset is None:
        if image is None:
            raise ValueError("You should pass either the image or the image_dataset!")
        image_dataset = FFImageDataset(
            images=image, no_patching=no_patching, patch_size=patch_size, overlap=overlap
        )
    # loop through the dataset and extract features
    dataloader = DataLoader(
        image_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"Start extracting features for {image_dataset.num_images} slices...")
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

    # # create the image dataset
    image_dataset = FFImageDataset(
        images=image, no_patching=no_patching, patch_size=patch_size, overlap=overlap
    )
    # create the storage
    storage = h5py.File(storage_path, "w")
    storage.attrs["num_slices"] = image_dataset.num_images
    storage.attrs["img_height"] = image_dataset.image_shape[0]
    storage.attrs["img_width"] = image_dataset.image_shape[1]
    storage.attrs["model"] = model_adapter.name
    storage.attrs["no_patching"] = no_patching
    storage.attrs["patch_size"] = patch_size
    storage.attrs["overlap"] = overlap

    for img_features, idx, total in extract_embeddings(
        model_adapter, image_dataset=image_dataset
    ):
        if storage.get(str(idx)) is None:
            # create a group for the slice
            grp = storage.create_group(str(idx))
            ds = grp.create_dataset(
                name="features",
                shape=img_features.shape,
                maxshape=(None,) + img_features.shape[1:],
                chunks=(1,) + img_features.shape[1:],
                dtype=np.float16,
                compression="lzf",
            )
            ds[:] = img_features
        else:
            # resize the dataset and append features to the slice/image group
            ds: h5py.Dataset = storage[str(idx)]["features"]  # type: ignore
            old_num_patches = ds.shape[0]
            num_patches = ds.shape[0] + img_features.shape[0]
            ds.resize(num_patches, axis=0)
            ds[old_num_patches:, ...] = img_features

        yield idx, total

    storage.close()
