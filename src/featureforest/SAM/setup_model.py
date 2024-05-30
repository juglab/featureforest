from pathlib import Path

import torch

from .models import MobileSAM


def setup_mobile_sam_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # sam model (light hq sam)
    model = MobileSAM.setup_model().to(device)
    # load weights
    weights = torch.load(
        Path(__file__).parent.joinpath(
            "./models/weights/mobile_sam.pt"
        ),
        map_location=device
    )
    model.load_state_dict(weights, strict=True)
    model.eval()

    return model, device
