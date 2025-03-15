import torch

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry

from featureforest.utils.downloader import download_model
from .adapter import SAMAdapter


MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.1/files/vit_b.pt",
    "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
}

MODEL_FNAMES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_b_lm": "vit_b_lm.pt",
    "vit_b_em_organelles": "vit_b_em_organelles.pt",
}

MODEL_NAMES = {
    "vit_h": "SAM",
    "vit_b_lm": "μSAM_LM",
    "vit_b_em_organelles": "μSAM_EM_Organelles",
}


def get_model(
    img_height: float, img_width: float, model_type: str = "vit_h", *args, **kwargs
) -> SAMAdapter:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # download model's weights
    model_url = MODEL_URLS[model_type]
    model_file = download_model(
        model_url=model_url,
        model_name=MODEL_FNAMES[model_type]
    )
    if model_file is None:
        raise ValueError(f"Could not download the model from {model_url}.")

    # init the model
    model: Sam = sam_model_registry[model_type[:5]](checkpoint=model_file)
    # to save some GPU memory, only put the encoder part on GPU
    sam_image_encoder = model.image_encoder.to(device)
    sam_image_encoder.eval()

    # create the model adapter
    sam_model_adapter = SAMAdapter(
        image_encoder=sam_image_encoder,
        img_height=img_height,
        img_width=img_width,
        device=device,
        name=MODEL_NAMES[model_type],
    )

    return sam_model_adapter
