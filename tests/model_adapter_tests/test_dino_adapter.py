import numpy as np
import pytest
import torch
import torch.nn as nn

from featureforest.models.DinoV2 import DinoV2Adapter, get_model
from featureforest.utils.data import get_stack_dims
from embedding_extraction import check_embedding_extraction


class MockDinoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino_patch_size = 14
        self.dino_out_channels = 384
        self.height = 70
        self.width = 70

    def get_intermediate_layers(self, x, *args, **kwargs):
        batch_size = x.shape[0]
        output = torch.ones(batch_size, self.dino_out_channels, self.height, self.width)
        return output, None


def get_mock_model(img_height: float, img_width: float) -> DinoV2Adapter:
    model = MockDinoEncoder()
    device = torch.device("cpu")
    dino_model_adapter = DinoV2Adapter(model, img_height, img_width, device)
    return dino_model_adapter


@pytest.mark.slow()
@pytest.mark.parametrize(
    "test_patch",
    [
        torch.ones((1, 3, 128, 128)),
        torch.ones((3, 3, 128, 128)),
        torch.ones((1, 3, 256, 256)),
        torch.ones((1, 3, 512, 512)),
    ],
)
def test_mock_adapter(test_patch: np.ndarray):
    img_height, img_width = test_patch.shape[-2:]
    real_adapter = get_model(img_height, img_width)
    mock_adapter = get_mock_model(img_height, img_width)

    transformed_input_patch_real = real_adapter.input_transforms(test_patch)
    transformed_input_patch_mock = mock_adapter.input_transforms(test_patch)

    result_real = real_adapter.model.get_intermediate_layers(
        transformed_input_patch_real, 1, return_class_token=False, reshape=True
    )[0]
    mock_result = mock_adapter.model.get_intermediate_layers(
        transformed_input_patch_mock
    )[0]

    assert len(result_real) == len(mock_result)
    assert result_real[0].shape == mock_result[0].shape


@pytest.mark.parametrize(
    "test_image, expected_output_shape, expected_slices",
    [
        (np.ones((256, 256)), (49, 42, 42, 384), 1),  # 2D
        (np.ones((256, 256, 3)), (49, 42, 42, 384), 1),  # 2D RGB
        (np.ones((2, 256, 256)), (49, 42, 42, 384), 2),  # 3D
        (np.ones((2, 256, 256, 3)), (49, 42, 42, 384), 2),  # 3D RGB
    ],
)
def test_dino_embedding_extraction(
    test_image: np.ndarray, expected_output_shape: tuple, expected_slices: int
):
    num_slices, img_height, img_width = get_stack_dims(test_image)
    model_adapter = get_mock_model(img_height, img_width)
    check_embedding_extraction(
        test_image, model_adapter, expected_output_shape, expected_slices
    )
