import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import v2 as tv_transforms2

from featureforest.models.base import BaseModelAdapter
from featureforest.utils.data import (
    get_nonoverlapped_patches,
)


class MobileSAMAdapter(BaseModelAdapter):
    """MobileSAM model adapter"""

    def __init__(
        self, model: nn.Module, img_height: int, img_width: int, device: torch.device
    ) -> None:
        super().__init__(model, img_height, img_width, device)
        self.name = "MobileSAM"
        # we need sam image encoder part
        self.encoder = self.model.image_encoder
        self.encoder_num_channels = 256
        self.embed_layer_num_channels = 64
        self._set_patch_size()
        self.device = device

        # input transform for sam
        self.sam_input_dim = 1024
        self.input_transforms = tv_transforms2.Compose(
            [
                tv_transforms2.Resize(
                    (self.sam_input_dim, self.sam_input_dim),
                    interpolation=tv_transforms2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )
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
        # get the mobile-sam encoder and embedding layer outputs
        with torch.no_grad():
            output, embed_output, _ = self.encoder(self.input_transforms(in_patches))

        # get non-overlapped feature patches
        out_feature_patches = get_nonoverlapped_patches(
            self.embedding_transform(output.cpu()), self.patch_size, self.overlap
        )
        embed_feature_patches = get_nonoverlapped_patches(
            self.embedding_transform(embed_output.cpu()), self.patch_size, self.overlap
        )
        # concat both features together on channel dimension
        output = torch.cat([out_feature_patches, embed_feature_patches], dim=-1)

        return output

    def get_total_output_channels(self) -> int:
        return self.encoder_num_channels + self.embed_layer_num_channels
