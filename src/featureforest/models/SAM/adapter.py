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


class SAMAdapter(BaseModelAdapter):
    """SAM model adapter (vit_h)
    """
    def __init__(
        self,
        image_encoder: nn.Module,
        img_height: float,
        img_width: float,
        device: torch.device
    ) -> None:
        super().__init__(image_encoder, img_height, img_width, device)
        self.name = "SAM"
        # we need sam image encoder part
        self.encoder = image_encoder
        self.encoder_num_channels = 256
        self.embed_layer_num_channels = 1280
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
        # get the mobile-sam encoder and embedding layer outputs
        with torch.no_grad():
            # output: b,256,64,64
            output = self.encoder(
                self.input_transforms(in_patches)
            )
            # embed_output: b,64,64,1280 -> b,1280,64,64
            embed_output = self.encoder.patch_embed(
                self.input_transforms(in_patches)
            ).permute(0, 3, 1, 2)

        # get non-overlapped feature patches
        out_feature_patches = get_nonoverlapped_patches(
            self.embedding_transform(output.cpu()),
            self.patch_size, self.overlap
        )
        embed_feature_patches = get_nonoverlapped_patches(
            self.embedding_transform(embed_output.cpu()),
            self.patch_size, self.overlap
        )

        return out_feature_patches, embed_feature_patches

    def get_total_output_channels(self) -> int:
        return self.encoder_num_channels + self.embed_layer_num_channels
