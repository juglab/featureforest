import numpy as np
import pytest
import torch
import torch.nn as nn

from featureforest.models.MobileSAM import MobileSAMAdapter, get_model
from featureforest.utils.data import get_stack_dims
from embedding_extraction import check_embedding_extraction


class MockMobileSAMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = self.mock_encode
        self.encoder_num_channels = 256
        self.embed_layer_num_channels = 64

    def mock_encode(self, x):
        batch_size = x.shape[0]
        output = torch.ones(
            batch_size,
            self.encoder_num_channels,
            self.embed_layer_num_channels,
            self.embed_layer_num_channels,
        )
        embed_output = torch.ones(
            batch_size,
            self.embed_layer_num_channels,
            self.encoder_num_channels,
            self.encoder_num_channels,
        )
        return output, embed_output, None


def get_mock_model(img_height: float, img_width: float) -> MobileSAMAdapter:
    model = MockMobileSAMEncoder()
    device = torch.device("cpu")
    sam_model_adapter = MobileSAMAdapter(model, img_height, img_width, device)
    return sam_model_adapter


@pytest.mark.slow()
@pytest.mark.parametrize(
    "test_patch",
    [
        torch.ones((1, 3, 128, 128)),
        torch.ones((3, 3, 128, 128)),
        torch.ones((8, 3, 128, 128)),
        torch.ones((8, 3, 256, 256)),
        torch.ones((8, 3, 512, 512))
    ],
)
def test_mock_adapter(test_patch: np.ndarray):
    real_adapter = get_model(512, 512)
    mock_adapter = get_mock_model(512, 512)

    transformed_input_patch_real = real_adapter.input_transforms(test_patch)
    transformed_input_patch_mock = mock_adapter.input_transforms(test_patch)

    result_real = real_adapter.encoder(transformed_input_patch_real)
    mock_result = mock_adapter.encoder(transformed_input_patch_mock)

    assert len(result_real) == len(mock_result)
    assert result_real[0].shape == mock_result[0].shape
    assert result_real[1].shape == mock_result[1].shape


@pytest.mark.parametrize(
    "test_image, expected_output_shape, expected_slices",
    [
        (np.ones((256, 256)), (9, 128, 128, 320), 1),  # 2D
        (np.ones((256, 256, 3)), (9, 128, 128, 320), 1),  # 2D RGB
        (np.ones((2, 256, 256)), (9, 128, 128, 320), 2),  # 3D
        (np.ones((2, 256, 256, 3)), (9, 128, 128, 320), 2)  # 3D RGB
    ],
)
def test_mobilesam_embedding_extraction(
    test_image: np.ndarray, expected_output_shape: tuple, expected_slices: int
):
    num_slices, img_height, img_width = get_stack_dims(test_image)
    model_adapter = get_mock_model(img_height, img_width)
    check_embedding_extraction(
        test_image, model_adapter, expected_output_shape, expected_slices
    )
