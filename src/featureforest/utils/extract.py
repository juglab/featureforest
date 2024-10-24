from typing import Generator, Tuple

import h5py
import numpy as np
import torch
from napari.utils import progress as np_progress

from featureforest.models import BaseModelAdapter
from featureforest.models.SAM import SAMAdapter
from featureforest.utils.data import get_stack_dims
from featureforest.utils.data import (
    patchify,
    get_stride_margin,
    is_image_rgb,
)


def get_slice_features(
    image: np.ndarray,
    patch_size: int,
    overlap: int,
    model_adapter: BaseModelAdapter,
    storage_group: h5py.Group,
) -> Generator[int, None, None]:
    """Extract the model features for one slice and save them into storage file.

    Args:
        image (np.ndarray): _description_
        patch_size (int): _description_
        overlap (int): _description_
        model_adapter (BaseModelAdapter): _description_
        storage_group (h5py.Group): _description_
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
    data_patches = patchify(img_data, patch_size, overlap)
    num_patches = len(data_patches)

    # set a low batch size
    batch_size = 8
    # for big SAM we need even lower batch size :(
    if isinstance(model_adapter, SAMAdapter):
        batch_size = 2

    num_batches = int(np.ceil(num_patches / batch_size))
    # prepare storage for the slice embeddings
    total_channels = model_adapter.get_total_output_channels()
    stride, _ = get_stride_margin(patch_size, overlap)
    dataset = storage_group.create_dataset(
        model_adapter.name, shape=(num_patches, stride, stride, total_channels),
        dtype=np.float16
    )

    # get sam encoder output for image patches
    print("extracting features:")
    for b_idx in np_progress(range(num_batches), desc="extracting feature"):
        print(f"batch #{b_idx + 1} of {num_batches}")
        start = b_idx * batch_size
        end = start + batch_size
        slice_features = model_adapter.get_features_patches(
            data_patches[start:end].to(model_adapter.device)
        )
        if not isinstance(slice_features, tuple):
            # model has only one output
            num_out = slice_features.shape[0]  # to take care of the last batch size
            dataset[start : start + num_out] = slice_features.to(torch.float16)
        else:
            # model has more than one output: put them into storage one by one
            ch_start = 0
            for feat in slice_features:
                num_out = feat.shape[0]
                ch_end = ch_start + feat.shape[-1]  # number of features
                dataset[start : start + num_out, :, :, ch_start:ch_end] = feat.to(torch.float16)
                ch_start = ch_end
        yield b_idx


def extract_embeddings_to_file(
    image: np.ndarray,
    storage_file_path: str,
    model_adapter: BaseModelAdapter
) -> Generator[Tuple[int, int], None, None]:
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap

    with h5py.File(storage_file_path, "w") as storage:
        num_slices, img_height, img_width = get_stack_dims(image)

        storage.attrs["num_slices"] = num_slices
        storage.attrs["img_height"] = img_height
        storage.attrs["img_width"] = img_width
        storage.attrs["model"] = model_adapter.name
        storage.attrs["patch_size"] = patch_size
        storage.attrs["overlap"] = overlap

        for slice_index in np_progress(
            range(num_slices), desc="extract features for slices"
        ):
            print(f"\nslice index: {slice_index}")
            slice_img = image[slice_index] if num_slices > 1 else image
            slice_grp = storage.create_group(str(slice_index))
            for _ in get_slice_features(
                slice_img, patch_size, overlap, model_adapter, slice_grp
            ):
                pass

            yield slice_index, num_slices
