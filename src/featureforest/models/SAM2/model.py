import torch

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

from featureforest.utils.downloader import download_model
from featureforest.models.SAM2.adapter import SAM2Adapter


MODEL_URLS = {
    "hvit_t": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "hvit_s": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "hvit_b": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "hvit_l": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

MODEL_FNAMES = {
    "hvit_t": "sam2.1_hiera_tiny.pt",
    "hvit_s": "sam2.1_hiera_small.pt",
    "hvit_b": "sam2.1_hiera_base_plus.pt",
    "hvit_l": "sam2.1_hiera_large.pt",
}

MODEL_NAMES = {
    "hvit_t": "SAM2_Tiny",
    "hvit_s": "SAM2_Small",
    "hvit_b": "SAM2_Base",
    "hvit_l": "SAM2_Large",
}

MODEL_CONFIG = {
    "hvit_t": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "hvit_s": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "hvit_b": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "hvit_l": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def get_model(
    img_height: float, img_width: float, model_type: str = "hvit_l", *args, **kwargs
) -> SAM2Adapter:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on '{device}'")

    # download model's weights
    model_url = MODEL_URLS[model_type]
    model_file = download_model(model_url=model_url, model_name=MODEL_FNAMES[model_type])
    if model_file is None:
        raise ValueError(f"Could not download the model from {model_url}.")

    # init the model
    model: SAM2Base = build_sam2(config_file=MODEL_CONFIG[model_type], ckpt_path=model_file, device="cpu")

    # to save some GPU memory, only put the encoder part on GPU
    sam_image_encoder = model.image_encoder.to(device)
    sam_image_encoder.eval()

    # create the model adapter
    sam2_model_adapter = SAM2Adapter(
        image_encoder=sam_image_encoder,
        img_height=img_height,
        img_width=img_width,
        device=device,
        name=MODEL_NAMES[model_type]
    )

    return sam2_model_adapter
