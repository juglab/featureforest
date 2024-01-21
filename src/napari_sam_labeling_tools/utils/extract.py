from napari.utils import progress as np_progress

import numpy as np
import torch
from torchvision import transforms

from .data import (
    DATA_PATCH_SIZE, TARGET_PATCH_SIZE,
    patchify, get_target_patches
)
from napari_sam_labeling_tools.SAM import sam_transform


def get_sam_embeddings_for_slice(sam_encoder, device, image):
    """get sam encoder features for one slice."""
    img_height, img_width = image.shape
    # image to torch tensor
    img_data = torch.from_numpy(image).to(torch.float32) / 255.0
    # for sam it should be 4D: BxCxHxW ; an RGB image.
    img_data = img_data.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
    # get data patches
    data_patches = patchify(img_data, DATA_PATCH_SIZE, TARGET_PATCH_SIZE)
    num_patches = len(data_patches)
    batch_size = 10
    num_batches = int(np.ceil(num_patches / batch_size))
    embedding_patches = []
    whole_img_patches = []
    # get embeddings
    with torch.no_grad():
        # get sam patch embeddings
        for b_idx in np_progress(range(num_batches), desc="getting patch embeddings"):
            print(f"batch #{b_idx + 1} of {num_batches}")
            start = b_idx * batch_size
            end = start + batch_size
            _, patch_embeddings, _ = sam_encoder(
                sam_transform(data_patches[start: end]).to(device)
            )
            embedding_patches.append(patch_embeddings.cpu())
        embedding_patches = torch.concat(embedding_patches, dim=0)  # shape: Nx64x256x256
        # get target patches
        # shape will be: B, patch_rows, patch_cols, target_size, target_size, C
        embedding_patches = get_target_patches(
            embedding_patches, img_height, img_width,
            DATA_PATCH_SIZE, TARGET_PATCH_SIZE
        )[0]  # remove batch size (B=1)

        # get sam encoder output for the whole slice
        resize_to_data = transforms.Resize((img_height, img_width), antialias=True)
        for _ in np_progress(range(1), desc="getting whole image embeddings"):
            print("getting whole slice embeddings")
            embeddings, _, _ = sam_encoder(
                sam_transform(img_data).to(device)
            )
            # shape: Nx256x256x256
            whole_img_patches = patchify(
                resize_to_data(embeddings.cpu()), DATA_PATCH_SIZE, TARGET_PATCH_SIZE
            )
            whole_img_patches = get_target_patches(
                whole_img_patches, img_height, img_width,
                DATA_PATCH_SIZE, TARGET_PATCH_SIZE
            )[0]  # remove batch size (B=1)

    # concat whole image embeddings with patch embeddings
    # final shape: patch_rows x patch_cols x target_size, target_size x C
    sam_embeddings = torch.concat(
        (embedding_patches, whole_img_patches),
        dim=-1
    ).numpy()

    return sam_embeddings
