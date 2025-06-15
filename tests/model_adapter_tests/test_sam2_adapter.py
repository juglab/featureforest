import numpy as np
import pytest
import torch
import torch.nn as nn
from embedding_extraction import check_embedding_extraction

from featureforest.models.SAM2.adapter import SAM2Adapter
from featureforest.utils.data import get_stack_dims


class MockSAM2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_num_channels = 256 * 3

    def __call__(self, x):
        batch_size = x.shape[0]
        # Mock the backbone_fpn output with 3 feature levels
        # [b, 256, 256, 256]
        # [b, 256, 128, 128]
        # [b, 256, 64, 64]
        level1 = torch.ones(batch_size, 256, 256, 256)
        level2 = torch.ones(batch_size, 256, 128, 128)
        level3 = torch.ones(batch_size, 256, 64, 64)

        return {"backbone_fpn": [level1, level2, level3]}


def get_mock_model(img_height: int, img_width: int) -> SAM2Adapter:
    model = MockSAM2Encoder()
    device = torch.device("cpu")
    sam2_model_adapter = SAM2Adapter(model, img_height, img_width, device)
    return sam2_model_adapter


@pytest.mark.parametrize(
    "img_height, img_width, expected_patch_size",
    [
        (
            512,
            512,
            256,
        ),  # SAM2 adapter seems to use 256 as default patch size for smaller images
        (256, 256, 256),
        (1024, 1024, 512),  # For larger images, it uses 512
    ],
)
def test_initialize_sam2_adapter(img_height, img_width, expected_patch_size):
    """Test that SAM2Adapter initializes correctly with different image sizes."""
    model = MockSAM2Encoder()
    device = torch.device("cpu")

    adapter = SAM2Adapter(model, img_height, img_width, device)

    assert adapter.name == "SAM2_Large"
    assert adapter.img_height == img_height
    assert adapter.img_width == img_width
    assert adapter.device == device
    assert adapter.encoder == model
    assert adapter.encoder_num_channels == 256 * 3
    assert adapter.patch_size == expected_patch_size
    assert adapter.overlap == expected_patch_size // 4
    assert adapter.sam_input_dim == 1024


@pytest.mark.parametrize(
    "test_patch",
    [
        torch.ones((1, 3, 128, 128)),
        torch.ones((3, 3, 128, 128)),
        torch.ones((1, 3, 256, 256)),
        torch.ones((1, 3, 512, 512)),
    ],
)
def test_process_input_patches(test_patch: torch.Tensor):
    """Test that SAM2Adapter processes input patches correctly."""
    img_height, img_width = test_patch.shape[-2:]
    adapter = get_mock_model(img_height, img_width)

    # Process the input patches
    output_features = adapter.get_features_patches(test_patch)

    # Check output shape
    batch_size = test_patch.shape[0]
    assert output_features.shape[0] == batch_size
    # The output should be in format [batch, height, width, channels]
    assert output_features.shape[3] == adapter.encoder_num_channels
    # Check that the output is a tensor
    assert isinstance(output_features, torch.Tensor)


def test_get_total_output_channels():
    """Test that SAM2Adapter returns the correct number of output channels."""
    adapter = get_mock_model(512, 512)

    # Check that the output channels is 256 * 3 = 768
    assert adapter.get_total_output_channels() == 256 * 3


@pytest.mark.parametrize(
    "test_image, expected_output_shape, expected_slices",
    [
        (np.ones((256, 256)), (4, 192, 192, 768), 1),  # 2D
        (np.ones((256, 256, 3)), (4, 192, 192, 768), 1),  # 2D RGB
    ],
)
def test_sam2_embedding_extraction_2d(
    test_image: np.ndarray, expected_output_shape: tuple, expected_slices: int
):
    """Test that SAM2Adapter extracts embeddings from 2D images correctly."""
    num_slices, img_height, img_width = get_stack_dims(test_image)
    model_adapter = get_mock_model(img_height, img_width)
    check_embedding_extraction(
        test_image, model_adapter, expected_output_shape, expected_slices
    )


@pytest.mark.parametrize(
    "test_image, expected_output_shape, expected_slices",
    [
        (np.ones((2, 256, 256)), (4, 192, 192, 768), 2),  # 3D
        (np.ones((2, 256, 256, 3)), (4, 192, 192, 768), 2),  # 3D RGB
    ],
)
def test_sam2_embedding_extraction_3d(
    test_image: np.ndarray, expected_output_shape: tuple, expected_slices: int
):
    """Test that SAM2Adapter extracts embeddings from 3D stacks correctly."""
    num_slices, img_height, img_width = get_stack_dims(test_image)
    model_adapter = get_mock_model(img_height, img_width)
    check_embedding_extraction(
        test_image, model_adapter, expected_output_shape, expected_slices
    )


def test_no_patching_mode():
    """Test that SAM2Adapter handles no_patching mode correctly."""
    img_height, img_width = 256, 256
    adapter = get_mock_model(img_height, img_width)

    # Default mode
    assert adapter.no_patching is False
    assert adapter.patch_size == 256
    assert adapter.overlap == 64

    # Enable no_patching mode
    adapter.no_patching = True
    assert adapter.no_patching is True
    assert adapter.patch_size == img_height
    assert adapter.overlap == 0

    # Disable no_patching mode
    adapter.no_patching = False
    assert adapter.no_patching is False
    assert adapter.patch_size == 256
    assert adapter.overlap == 64
