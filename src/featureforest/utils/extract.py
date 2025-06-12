from collections.abc import Generator
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from napari.utils import progress as np_progress
from torch.utils.data import DataLoader

from featureforest.models import BaseModelAdapter
from featureforest.models.SAM import SAMAdapter
from featureforest.utils.data import (
    get_stride_margin,
    is_image_rgb,
    patchify,
)
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


def get_dataset(
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


def get_slice_features(
    image: np.ndarray,
    model_adapter: BaseModelAdapter,
    storage_group: Optional[h5py.Group] = None,
) -> Generator[Union[int, tuple[int, np.ndarray]], None, None]:
    """Extract features for one image/slice using the given model adapter and
    save them into a storage file or yield them batch by batch.

    Args:
        image (np.ndarray): Input image
        model_adapter (BaseModelAdapter): Model adapter to extract features from
        storage_group (Optional[h5py.Group]): h5 file group where the extracted features
        will be saved. If None, will yield patch features batch by batch.
    Returns:
        Generator[Union[int, tuple[int, np.ndarray]]]: A generator yielding either the
        current batch number or a tuple containing the current batch number
        and the corresponding patch features.
    """
    # image to torch tensor
    img_data = torch.from_numpy(image.copy()).to(torch.float32)
    # normalize in [0, 1]
    _min = img_data.min()
    _max = img_data.max()
    img_data = (img_data - _min) / (_max - _min)
    # for sam the input image should be 4D: BxCxHxW ; an RGB image.
    if is_image_rgb(image):
        # it's already RGB, put the channels first and add a batch dim.
        img_data = img_data[..., :3]  # ignore the Alpha channel (in case of PNG).
        img_data = img_data.permute([2, 0, 1]).unsqueeze(0)
    else:
        img_data = img_data.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

    # get input patches
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    data_patches = patchify(img_data, patch_size, overlap)
    num_patches = len(data_patches)

    # set a low batch size
    batch_size = 8
    # for big SAM we need even lower batch size :(
    if isinstance(model_adapter, SAMAdapter):
        batch_size = 2

    num_batches = int(np.ceil(num_patches / batch_size))
    # prepare storage for the image embeddings
    if storage_group is not None:
        total_channels = model_adapter.get_total_output_channels()
        stride, _ = get_stride_margin(patch_size, overlap)
        dataset = storage_group.create_dataset(
            model_adapter.name,
            shape=(num_patches, stride, stride, total_channels),
            dtype=np.float16,
            compression="lzf",
        )

    # get sam encoder output for image patches
    print("extracting features:")
    for b_idx in np_progress(range(num_batches), desc="extracting features"):
        print(f"batch #{b_idx + 1} of {num_batches}")
        start = b_idx * batch_size
        end = start + batch_size
        slice_features = model_adapter.get_features_patches(
            data_patches[start:end].to(model_adapter.device)
        )
        if isinstance(slice_features, tuple):  # model with more than one output
            slice_features = torch.cat(slice_features, dim=-1)
        if storage_group is not None:
            # to take care of the last batch size that might be smaller than batch_size
            num_out = slice_features.shape[0]
            dataset[start : start + num_out] = (
                slice_features.to(torch.float16).cpu().numpy()
            )
            yield b_idx
        else:
            yield b_idx, slice_features.numpy()


def extract_embeddings_to_file(
    image: np.ndarray | str, storage_path: str, model_adapter: BaseModelAdapter
) -> Generator[tuple[int, int], None, None]:
    no_patching = model_adapter.no_patching
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    batch_size = get_batch_size(model_adapter)

    dataset = get_dataset(
        image=image, no_patching=no_patching, patch_size=patch_size, overlap=overlap
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for img_data, indices in dataloader:
        print(f"images: {img_data.shape}\nslices: {indices}")
        features = model_adapter.get_features_patches(img_data.to(model_adapter.device))
        print(f"features shape: {features.shape}")
        unique_slices = torch.unique(indices[:, 0]).numpy()
        print(f"unique slices: {unique_slices}")
        for idx in unique_slices:
            img_features = features[indices[:, 0] == idx].numpy().astype(np.float16)
            print(f"image: {idx}, features shape: {img_features.shape}")
            yield idx, len(unique_slices)

    # with h5py.File(storage_path, "w") as storage:
    #     num_slices, img_height, img_width = get_stack_dims(image)

    #     storage.attrs["num_slices"] = num_slices
    #     storage.attrs["img_height"] = img_height
    #     storage.attrs["img_width"] = img_width
    #     storage.attrs["model"] = model_adapter.name
    #     storage.attrs["patch_size"] = patch_size
    #     storage.attrs["overlap"] = overlap

    #     for slice_index in np_progress(
    #         range(num_slices), desc="extract features for slices"
    #     ):
    #         print(f"\nslice index: {slice_index}")
    #         slice_img = image[slice_index].copy() if num_slices > 1 else image.copy()
    #         slice_img = image_to_uint8(slice_img)  # image must be an uint8 array
    #         slice_grp = storage.create_group(str(slice_index))
    #         for _ in get_slice_features(slice_img, model_adapter, slice_grp):
    #             pass

    #         yield slice_index, num_slices
