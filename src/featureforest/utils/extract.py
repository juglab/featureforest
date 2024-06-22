from napari.utils import progress as np_progress

import numpy as np
import h5py
import torch

from .data import (
    patchify, get_stride_margin,
    is_image_rgb,
)
from featureforest.models.base import BaseModelAdapter


def get_slice_features(
    image: np.ndarray,
    patch_size: int,
    overlap: int,
    model_adapter: BaseModelAdapter,
    device: torch.device,
    storage_group: h5py.Group
) -> None:
    """Extract the model features for one slice and save them into storage file.

    Args:
        image (np.ndarray): _description_
        patch_size (int): _description_
        overlap (int): _description_
        model_adapter (BaseModelAdapter): _description_
        device (torch.device): _description_
        storage_group (h5py.Group): _description_
    """
    img_height, img_width = image.shape[:2]
    # image to torch tensor
    img_data = torch.from_numpy(image).to(torch.float32) / 255.0
    # for sam the input image should be 4D: BxCxHxW ; an RGB image.
    if is_image_rgb(image):
        # it's already RGB, put the channels first and add a batch dim.
        img_data = img_data.permute([2, 0, 1]).unsqueeze(0)
    else:
        img_data = img_data.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

    # get input patches
    data_patches = patchify(img_data, patch_size, overlap)
    num_patches = len(data_patches)
    batch_size = 10
    num_batches = int(np.ceil(num_patches / batch_size))
    # prepare storage for the slice embeddings
    total_channels = model_adapter.get_total_output_channels()
    stride, _ = get_stride_margin(patch_size, overlap)
    dataset = storage_group.create_dataset(
        model_adapter.name, shape=(
            num_patches, stride, stride, total_channels
        )
    )

    # get sam encoder output for image patches
    print("\nextracting slice features:")
    for b_idx in np_progress(
        range(num_batches), desc="extracting slice feature:"
    ):
        print(f"batch #{b_idx + 1} of {num_batches}")
        start = b_idx * batch_size
        end = start + batch_size
        slice_features = model_adapter.get_features_patches(
            data_patches[start: end].to(device)
        )
        if not isinstance(slice_features, tuple):
            # model has only one output
            num_out = slice_features.shape[0]  # to take care of the last batch size
            dataset[start: start + num_out] = slice_features
        else:
            # model has more than one output: put them into storage one by one
            ch_start = 0
            for feat in slice_features:
                num_out = feat.shape[0]
                ch_end = ch_start + feat.shape[-1]  # number of features
                dataset[start: start + num_out, :, :, ch_start: ch_end] = feat
                ch_start = ch_end
