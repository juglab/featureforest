from pathlib import Path

import torch

from .models import LightHQSAM


def setup_lighthq_sam_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # sam model (light hq sam)
    model = LightHQSAM.setup().to(device)
    # load weights
    weights = torch.load(
        Path(__file__).parent.joinpath(
            "./models/weights/sam_hq_vit_tiny.pth"
        ),
        map_location=device
    )
    model.load_state_dict(weights, strict=True)
    model.eval()

    return model, device
