from napari.utils import progress as np_progress

import numpy as np
import h5py
import torch
from torchvision import transforms

from .data import (
    patchify, get_target_patches,
    is_image_rgb,
)
from featureforest.SAM import (
    ENCODER_OUT_CHANNELS, EMBED_PATCH_CHANNELS,
    sam_transform
)


def get_patch_sizes(img_height, img_width):
    patch_size = 512
    target_patch_size = 128
    img_min_dim = min(img_height, img_width)
    while img_min_dim - patch_size < 100:
        patch_size = patch_size // 2
    if patch_size < 256:
        target_patch_size = patch_size // 2

    return patch_size, target_patch_size


def get_sam_embeddings_for_slice(
        image, patch_size, target_patch_size,
        sam_encoder, device, storage_group: h5py.Group
):
    """get sam features for one slice."""
    img_height, img_width = image.shape[:2]
    # image to torch tensor
    img_data = torch.from_numpy(image).to(torch.float32) / 255.0
    # for sam the input image should be 4D: BxCxHxW ; an RGB image.
    if is_image_rgb(image):
        # it's already RGB, put the channels first and add a batch dim.
        img_data = img_data.permute([2, 0, 1]).unsqueeze(0)
    else:
        img_data = img_data.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

    # to resize encoder output back to the input patch size
    embedding_transform = transforms.Compose([
        transforms.Resize(
            (patch_size, patch_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        ),
        # transforms.GaussianBlur(kernel_size=3, sigma=1.0)
    ])

    # get sam encoder output for image patches
    data_patches = patchify(img_data, patch_size, target_patch_size)
    num_patches = len(data_patches)
    batch_size = 10
    num_batches = int(np.ceil(num_patches / batch_size))
    # prepare storage for the slice embeddings
    total_channels = ENCODER_OUT_CHANNELS + EMBED_PATCH_CHANNELS
    dataset = storage_group.create_dataset(
        "sam", shape=(
            num_patches, target_patch_size, target_patch_size, total_channels
        )
    )

    with torch.no_grad():
        print("\ngetting SAM encoder & patch_embed output:")
        for b_idx in np_progress(
            range(num_batches), desc="getting SAM encoder & patch_embed outputs"
        ):
            print(f"batch #{b_idx + 1} of {num_batches}")
            start = b_idx * batch_size
            end = start + batch_size
            output, embed_output, _ = sam_encoder(
                sam_transform(data_patches[start: end]).to(device)
            )
            # output:          Bx256x64x64,    embed_output: Bx64x256x256
            # after transform: Bx256x512x512,  embed_output: Bx64x512x512
            # target patch:    B, target_size, target_size, C
            num_out = output.shape[0]
            dataset[
                start: start + num_out, :, :, :ENCODER_OUT_CHANNELS
            ] = get_target_patches(
                embedding_transform(output.cpu()),
                patch_size, target_patch_size
            )
            dataset[
                start: start + num_out, :, :, ENCODER_OUT_CHANNELS:
            ] = get_target_patches(
                embedding_transform(embed_output.cpu()),
                patch_size, target_patch_size
            )
