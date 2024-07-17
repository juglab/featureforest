from typing import Tuple

import torch

# from featureforest.utils.downloader import download_model
from .adapter import DinoV2Adapter


def get_model(
    img_height: float, img_width: float, *args, **kwargs
) -> DinoV2Adapter:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # get the pretrained model
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(device)
    # model_url =
    # "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
    model.eval()

    # create the model adapter
    dino_model_adapter = DinoV2Adapter(
        model, img_height, img_width, device
    )

    return dino_model_adapter
