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
        img_height: float,
        img_width: float,
        device: torch.device
    ) -> None:
        """Initialization function

        Args:
            model (nn.Module): the pytorch model (e.g. a ViT encoder)
            input_transforms (tv_transforms2.Compose):
                input transformations for the specific model
            img_height (float): input image height
            img_width (float): input image width
        """
        self.name = "Base"
        self.model = model
        self.img_height = img_height
        self.img_width = img_width
        self.device = device
        # set patch size and overlap
        self.patch_size = 512
        self.overlap = self.patch_size // 2
        # input image transforms
        self.input_transforms = tv_transforms2.Compose([
            tv_transforms2.Resize(
                (1024, 1024),
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
        """Sets the proper patch size and patch overlap
        with respect to the model & image resolution.
        """
        raise NotImplementedError

    def get_features_patches(
        self, in_patches: Tensor
    ) -> Tensor:
        """Returns model's extracted features.
        This is an abstract function, and should be overridden.

        Args:
            in_patches (Tensor): input patches

        Returns:
            Tensor: model's extracted features
        """
        # get the model output
        with torch.no_grad():
            output_features = self.model(self.input_transforms(in_patches))
        # assert self.patch_size == out_features.shape[-1]

        # get non-overlapped feature patches
        feature_patches = get_nonoverlapped_patches(
            self.embedding_transform(output_features.cpu()),
            self.patch_size, self.overlap
        )

        return feature_patches

    def get_total_output_channels(self) -> int:
        """Returns total number of model output channels (number of feature maps).

        Returns:
            int: total number of output channels
        """
        return 256
