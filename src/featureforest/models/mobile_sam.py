import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import v2 as tv_transforms2

from .base import BaseModelAdapter
from ..utils.data import (
    get_nonoverlapped_patches,
)


class MobileSAM(BaseModelAdapter):
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

    def get_features_patches(
        self, in_patches: Tensor
    ) -> Tensor:
        # get the model output
        with torch.no_grad():
            out_features = self.model(self.input_transforms(in_patches))
        # assert self.patch_size == out_features.shape[-1]

        # get non-overlapped feature patches
        feature_patches = get_nonoverlapped_patches(
            self.embedding_transform(out_features.cpu()),
            self.patch_size, self.overlap
        )

        return feature_patches
