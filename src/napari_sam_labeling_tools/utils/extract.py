import napari
import napari.utils.notifications as notif
from napari.utils.events import Event
from napari.qt.threading import create_worker
from napari.utils import progress as np_progress

import numpy as np
import torch
from torchvision import transforms

from .data import (
    DATA_PATCH_SIZE, TARGET_PATCH_SIZE,
    patchify, unpatchify
)
from napari_sam_labeling_tools.SAM import sam_transform


def get_sam_embeddings_for_slice(sam_encoder, device, image_layer, slice_index):
    """get sam encoder features for one slice."""
    img_depth, img_height, img_width = image_layer.data.shape
    # image to torch tensor
    img_data = torch.from_numpy(
        image_layer.data[slice_index]
    ).to(torch.float32) / 255.0
    # for sam it should be 4D: BxCxHxW ; an RGB image.
    img_data = img_data.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
    # get data patches
    data_patches = patchify(img_data, DATA_PATCH_SIZE, TARGET_PATCH_SIZE)
    num_patches = len(data_patches)
    batch_size = 10
    num_batches = int(np.ceil(num_patches / batch_size))
    data_embeddings = []
    whole_image_embbedings = []
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
            data_embeddings.append(patch_embeddings.cpu())
            # yield (b_idx, num_batches)
        # get sam encoder output for whole images
        for i in np_progress(range(1), desc="getting whole image embeddings"):
            print("getting whole slice embeddings")
            embeddings, _, _ = sam_encoder(
                sam_transform(img_data).to(device)
            )
            whole_image_embbedings.append(embeddings.cpu())
            # yield (i, 1)

    # resize whole image embeddings to the image size
    resize_to_data = transforms.Resize((img_height, img_width), antialias=True)
    whole_image_embbedings = resize_to_data(
        torch.concat(whole_image_embbedings, dim=0)
    )
    # unpatchify patch embeddings into image size
    data_embeddings = torch.concat(data_embeddings, dim=0)
    sam_embeddings = unpatchify(
        data_embeddings, img_height, img_width,
        DATA_PATCH_SIZE, TARGET_PATCH_SIZE
    )
    # concat whole image embeddings with patch embeddings
    sam_embeddings = torch.concat(
        (sam_embeddings, whole_image_embbedings),
        dim=1
    )
    # B, embedding_size, H, W
    sam_embeddings = sam_embeddings.permute(
        0, 2, 3, 1).contiguous().cpu().numpy()

    return sam_embeddings
