from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import v2 as tv_transforms2

from featureforest.models.base import BaseModelAdapter
from featureforest.utils.data import (
    get_patch_size,
    get_nonoverlapped_patches,
)


class SAM2Adapter(BaseModelAdapter):
    """SAM2 model adapter
    """
    def __init__(
        self,
        image_encoder: nn.Module,
        img_height: float,
        img_width: float,
        device: torch.device,
        name: str = "SAM2_Large"
    ) -> None:
        super().__init__(image_encoder, img_height, img_width, device)
        # for different flavors of SAM2 only the name is different.
        self.name = name
        # we need sam2 image encoder part
        self.encoder = image_encoder
        self.encoder_num_channels = 256 * 3
        self._set_patch_size()
        self.device = device

        # input transform for sam
        self.sam_input_dim = 1024
        self.input_transforms = tv_transforms2.Compose([
            tv_transforms2.Resize(
                (self.sam_input_dim, self.sam_input_dim),
                interpolation=tv_transforms2.InterpolationMode.BICUBIC,
                antialias=True
            ),
        ])
        # to transform feature patches back to the original patch size
        self.embedding_transform = tv_transforms2.Compose([
            tv_transforms2.Resize(
                (self.patch_size, self.patch_size),
                interpolation=tv_transforms2.InterpolationMode.BICUBIC,
                antialias=True
            ),
        ])

    def _set_patch_size(self) -> None:
        self.patch_size = get_patch_size(self.img_height, self.img_width)
        self.overlap = self.patch_size // 2

    def get_features_patches(
        self, in_patches: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # get the image encoder outputs
        with torch.no_grad():
            output = self.encoder(
                self.input_transforms(in_patches)
            )
        # backbone_fpn contains 3 levels of features from hight to low resolution.
        # [b, 256, 256, 256]
        # [b, 256, 128, 128]
        # [b, 256, 64, 64]
        features = [
            self.embedding_transform(feat.cpu())
            for feat in output["backbone_fpn"]
        ]
        features = torch.cat(features, dim=1)
        out_feature_patches = get_nonoverlapped_patches(features, self.patch_size, self.overlap)

        return out_feature_patches

    def get_total_output_channels(self) -> int:
        return 256 * 3
