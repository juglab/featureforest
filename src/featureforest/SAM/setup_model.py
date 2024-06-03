from pathlib import Path

import torch

from ..utils.downloader import download_model
from .models import MobileSAM


def setup_mobile_sam_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # sam model (light hq sam)
    model = MobileSAM.setup_model().to(device)
    # download model's weights
    model_file = download_model(
        model_url="https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        model_name="mobile_sam.pt"
    )
    # load weights
    weights = torch.load(model_file, map_location=device)
    model.load_state_dict(weights, strict=True)
    model.eval()

    return model, device
