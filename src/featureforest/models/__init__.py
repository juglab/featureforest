from typing import List
from functools import partial

from .base import BaseModelAdapter
from .MobileSAM import get_model as get_mobile_sam_model
from .SAM import get_model as get_sam_model
from .DinoV2 import get_model as get_dino_v2_model
from .SAM2 import get_large_model as get_sam2_large_model
from .SAM2 import get_base_model as get_sam2_base_model


_MODELS_DICT = {
    "SAM2_Large": get_sam2_large_model,
    "SAM2_Base": get_sam2_base_model,
    "MobileSAM": get_mobile_sam_model,
    "SAM": get_sam_model,
    "DinoV2": get_dino_v2_model,
    "μSAM_LM": partial(get_sam_model, model_type="vit_b_lm"),
    "μSAM_EM_Organelles": partial(get_sam_model, model_type="vit_b_em_organelles"),
}


def get_available_models() -> List[str]:
    return list(_MODELS_DICT.keys())


def get_model(
    model_name: str, img_height: float, img_width: float, *args, **kwargs
) -> BaseModelAdapter:
    """Returns the requested model adapter.

    Args:
        model_name (str): the model name
        img_height (float): input image height
        img_width (float): input image width

    Raises:
        ValueError: if the model was not found

    Returns:
        BaseModelAdapter: the model adapter
    """
    if model_name not in _MODELS_DICT:
        raise ValueError(f"Model {model_name} was not found!")

    return _MODELS_DICT[model_name](img_height, img_width, *args, **kwargs)
