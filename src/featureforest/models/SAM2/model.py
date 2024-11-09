from pathlib import Path

import torch

from sam2.modeling.sam2_base import SAM2Base
from sam2.build_sam import build_sam2

from featureforest.utils.downloader import download_model
from featureforest.models.SAM2.adapter import SAM2Adapter


def get_large_model(
    img_height: float, img_width: float, *args, **kwargs
) -> SAM2Adapter:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # download model's weights
    model_url = \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    model_file = download_model(
        model_url=model_url,
        model_name="sam2.1_hiera_large.pt"
    )
    if model_file is None:
        raise ValueError(f"Could not download the model from {model_url}.")

    # init the model
    model: SAM2Base = build_sam2(
        config_file= "configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path=model_file,
        device="cpu"
    )
    # to save some GPU memory, only put the encoder part on GPU
    sam_image_encoder = model.image_encoder.to(device)
    sam_image_encoder.eval()

    # create the model adapter
    sam2_model_adapter = SAM2Adapter(
        sam_image_encoder, img_height, img_width, device, "SAM2_Large"
    )

    return sam2_model_adapter


def get_base_model(
    img_height: float, img_width: float, *args, **kwargs
) -> SAM2Adapter:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # download model's weights
    model_url = \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    model_file = download_model(
        model_url=model_url,
        model_name="sam2.1_hiera_base_plus.pt"
    )
    if model_file is None:
        raise ValueError(f"Could not download the model from {model_url}.")

    # init the model
    model: SAM2Base = build_sam2(
        config_file= "configs/sam2.1/sam2.1_hiera_b+.yaml",
        ckpt_path=model_file,
        device="cpu"
    )
    # to save some GPU memory, only put the encoder part on GPU
    sam_image_encoder = model.image_encoder.to(device)
    sam_image_encoder.eval()

    # create the model adapter
    sam2_model_adapter = SAM2Adapter(
        sam_image_encoder, img_height, img_width, device, "SAM2_Base"
    )

    return sam2_model_adapter
