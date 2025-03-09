from typing import List
from functools import partial

from .base import BaseModelAdapter
from .SAM import get_model as get_sam_model
from .SAM2 import get_model as get_sam2_model
from .DinoV2 import get_model as get_dino_v2_model
from .MobileSAM import get_model as get_mobile_sam_model


_MODELS_DICT = {
    "SAM2_Tiny": partial(get_sam2_model, model_type="hvit_t"),
    "SAM2_Small": partial(get_sam2_model, model_type="hvit_s"),
    "SAM2_Base": partial(get_sam2_model, model_type="hvit_b"),
    "SAM2_Large": partial(get_sam2_model, model_type="hvit_l"),
    "MobileSAM": get_mobile_sam_model,
    "SAM_Base": partial(get_sam_model, model_type="vit_b"),
    "SAM_Large": partial(get_sam_model, model_type="vit_l"),
    "SAM_Huge": partial(get_sam_model, model_type="vit_h"),
    "μSAM_LM": partial(get_sam_model, model_type="vit_b_lm"),
    "μSAM_EM_Organelles": partial(get_sam_model, model_type="vit_b_em_organelles"),
    "DinoV2": get_dino_v2_model,
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
