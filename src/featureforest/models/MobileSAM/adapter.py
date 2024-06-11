from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import v2 as tv_transforms2

from featureforest.models.base import BaseModelAdapter
from featureforest.utils.data import (
    get_nonoverlapped_patches,
)


class MobileSAMAdapter(BaseModelAdapter):
    """MobileSAM model adapter
    """
    def __init__(
        self,
        model: nn.Module,
        input_transforms: tv_transforms2.Compose,
        patch_size: int,
        overlap: int,
    ) -> None:
        super().__init__(model, input_transforms, patch_size, overlap)
        # we need sam image encoder part
        self.encoder = self.model.image_encoder
        self.encoder_num_channels = 256
        self.embed_layer_num_channels = 64

    def get_features_patches(
        self, in_patches: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # get the mobile-sam encoder and embedding layer outputs
        with torch.no_grad():
            output, embed_output, _ = self.encoder(
                self.input_transforms(in_patches)
            )

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
