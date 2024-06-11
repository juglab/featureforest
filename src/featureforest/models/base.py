import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import v2 as tv_transforms2

from ..utils.data import (
    get_nonoverlapped_patches,
)


class BaseModelAdapter:
    """Base class for adapting any models in featureforest.
    """
    def __init__(
        self,
        model: nn.Module,
        input_transforms: tv_transforms2.Compose,
        patch_size: int,
        overlap: int,
    ) -> None:
        """Initialization function

        Args:
            model (nn.Module): the pytorch model (e.g. a ViT encoder)
            input_transforms (tv_transforms2.Compose): input transformations for the specific model
            patch_size (int): input patch size
            overlap (int): input patch overlap
        """
        self.model = model
        self.input_transforms = input_transforms
        self.patch_size = patch_size
        self.overlap = overlap
        # to transform feature patches to the original patch size
        self.embedding_transform = tv_transforms2.Compose([
            tv_transforms2.Resize(
                (self.patch_size, self.patch_size),
                interpolation=tv_transforms2.InterpolationMode.BICUBIC,
                antialias=True
            ),
        ])

    def get_features_patches(
        self, in_patches: Tensor
    ) -> Tensor:
        """Returns a tensor of model's extracted features.
        This function is more like an abstract function, and should be overridden.

        Args:
            in_patches (Tensor): input patches

        Returns:
            Tensor: model's extracted features
        """
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

    def get_total_output_channels(self) -> int:
        """Returns total number of model output channels (a.k.a. number of feature maps).

        Returns:
            int: total number of output channels
        """
        return 256
