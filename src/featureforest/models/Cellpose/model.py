from pathlib import Path

import torch

from featureforest.models.Cellpose.adapter import CellposeAdapter
from featureforest.models.Cellpose.cellpose_downsample import downsample


class Cyto3Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nbase = [2, 32, 64, 128, 256]
        self.sz = 3
        self.max_pool = True
        self.downsample = downsample(
            self.nbase, self.sz, conv_3D=False, max_pool=self.max_pool
        )
        # load downsample weights
        weight_file = Path(__file__).parent.joinpath("cyto3_downsample.pth")
        assert weight_file.exists(), "Couldn't find cellpose weights"
        self.downsample.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, x):
        out = self.downsample(x)
        return out


def get_model(img_height: int, img_width: int, *args, **kwargs) -> CellposeAdapter:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # init the model
    model = Cyto3Encoder().to(device)
    model.eval()

    # create the model adapter
    cellpose_adapter = CellposeAdapter(
        model, img_height, img_width, device, "Cellpose_cyto3"
    )

    return cellpose_adapter


if __name__ == "__main__":
    model = Cyto3Encoder()
    print(model)
