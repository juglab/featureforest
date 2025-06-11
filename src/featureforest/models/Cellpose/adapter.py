import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import v2 as tv_transforms2

from featureforest.models.base import BaseModelAdapter
from featureforest.utils.data import (
    get_nonoverlapped_patches,
)


class CellposeAdapter(BaseModelAdapter):
    """Cellpose model adapter"""

    def __init__(
        self,
        image_encoder: nn.Module,
        img_height: int,
        img_width: int,
        device: torch.device,
        name: str = "Cellpose_cyto3",
    ) -> None:
        super().__init__(image_encoder, img_height, img_width, device)
        # for different flavors of SAM2 only the name is different.
        self.name = name
        self.encoder = image_encoder
        # self.encoder_num_channels = 480
        self.device = device
        self._set_patch_size()
        assert int(self.patch_size / 4) == self.patch_size / 4, (
            f"patch size {self.patch_size} is not divisible by 4"
        )

        # input transform for sam
        self.input_transforms = None
        # to transform feature patches back to the original patch size
        self.embedding_transform = tv_transforms2.Compose(
            [
                tv_transforms2.Resize(
                    (self.patch_size, self.patch_size),
                    interpolation=tv_transforms2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )

    def get_features_patches(self, in_patches: Tensor) -> Tensor:
        # cellpose works on microscopic channels not RGB
        # we need to select one channel and add a second zero channel
        in_patches = torch.cat(
            [in_patches[:, 0:1], torch.zeros_like(in_patches[:, 0:1])], dim=1
        )
        # get the image encoder outputs
        with torch.no_grad():
            output = self.encoder(in_patches)
        # output has four levels of features from hight to low resolution.
        # [b, 32, p, p]
        # [b, 64, p/2, p/2]
        # [b, 128, p/4, p/4]
        # [b, 256, p/8, p/8]
        features = [self.embedding_transform(feat.cpu()) for feat in output]
        features = torch.cat(features, dim=1)
        out_feature_patches = get_nonoverlapped_patches(
            features, self.patch_size, self.overlap
        )

        return out_feature_patches

    def get_total_output_channels(self) -> int:
        return 32 + 64 + 128 + 256  # 480
