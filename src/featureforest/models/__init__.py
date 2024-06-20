from typing import Tuple, List

import torch

from .base import BaseModelAdapter
from .MobileSAM import get_model as get_mobile_sam_model
from .DinoV2 import get_model as get_dino_v2_model


_MODELS_DICT = {
    "MobileSAM": get_mobile_sam_model,
    "DinoV2": get_dino_v2_model,
}


def get_available_models() -> List[str]:
    return list(_MODELS_DICT.keys())


def get_model(
    model_name: str, img_height: float, img_width: float, *args, **kwargs
) -> Tuple[BaseModelAdapter, torch.device]:
    """Returns the requested model adapter.

    Args:
        model_name (str): the model name
        img_height (float): input image height
        img_width (float): input image width

    Raises:
        ValueError: if the model was not found

    Returns:
        Tuple[BaseModelAdapter, torch.device]: the model adapter and torch device
    """
    if model_name not in _MODELS_DICT:
        raise ValueError(f"Model {model_name} was not found!")

    return _MODELS_DICT[model_name](img_height, img_width, *args, **kwargs)
