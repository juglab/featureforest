from typing import Tuple

import torch

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry

from featureforest.utils.downloader import download_model
from .adapter import SAMAdapter


def get_model(
    img_height: float, img_width: float, *args, **kwargs
) -> SAMAdapter:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # download model's weights
    model_url = \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    model_file = download_model(
        model_url=model_url,
        model_name="sam_vit_h_4b8939.pth"
    )
    if model_file is None:
        raise ValueError(f"Could not download the model from {model_url}.")

    # init the model
    model: Sam = sam_model_registry["vit_h"](checkpoint=model_file)
    # to save some GPU memory, only put the encoder part on GPU
    sam_image_encoder = model.image_encoder.to(device)
    sam_image_encoder.eval()

    # create the model adapter
    sam_model_adapter = SAMAdapter(
        sam_image_encoder, img_height, img_width, device
    )

    return sam_model_adapter
