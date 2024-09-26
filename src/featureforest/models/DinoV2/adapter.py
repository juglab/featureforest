from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import v2 as tv_transforms2

from featureforest.models.base import BaseModelAdapter
from featureforest.utils.data import (
    get_nonoverlapped_patches,
)


class DinoV2Adapter(BaseModelAdapter):
    """DinoV2 model adapter
    """
    def __init__(
        self,
        model: nn.Module,
        img_height: float,
        img_width: float,
        device: torch.device
    ) -> None:
        super().__init__(model, img_height, img_width, device)
        self.name = "DinoV2"
        self.model = self.model
        self.dino_patch_size = 14
        self.dino_out_channels = 384
        self._set_patch_size()
        self.device = device

        # input transform for dinov2
        self.input_transforms = tv_transforms2.Compose([
            tv_transforms2.ToImage(),
            tv_transforms2.Resize(self.patch_size * self.dino_patch_size),
            tv_transforms2.ToDtype(dtype=torch.float32, scale=True)
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
        self.patch_size = self.dino_patch_size * 5
        self.overlap = self.dino_patch_size * 2

    def get_features_patches(
        self, in_patches: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # get the mobile-sam encoder and embedding layer outputs
        with torch.no_grad():
            # we use get_intermediate_layers method of dinov2, which returns a tuple.
            # output shape: b, 384, h, w if reshape is true.
            output_features = self.model.get_intermediate_layers(
                self.input_transforms(in_patches), 1,
                return_class_token=False, reshape=True
            )[0]

        # get non-overlapped feature patches
        feature_patches = get_nonoverlapped_patches(
            output_features.cpu(),
            self.patch_size, self.overlap
        )

        return feature_patches

    def get_total_output_channels(self) -> int:
        return self.dino_out_channels
