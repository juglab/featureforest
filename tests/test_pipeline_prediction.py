from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from featureforest.utils.pipeline_prediction import run_prediction_pipeline


@pytest.fixture
def mock_model_adapter():
    adapter = MagicMock()
    adapter.no_patching = False
    adapter.patch_size = 128
    adapter.overlap = 32
    adapter.get_total_output_channels.return_value = 320
    return adapter


@pytest.fixture
def mock_rf_model():
    rf = MagicMock()
    rf.predict.side_effect = lambda x: np.ones((x.shape[0],), dtype=np.uint8)
    return rf


@pytest.fixture
def mock_stack_dataset():
    dataset = MagicMock()
    dataset.image_shape = (240, 240)
    return dataset


@pytest.fixture
def mock_extract_embeddings():
    # Simulate two slices, each with two patches
    def generator(*args, **kwargs):
        yield np.zeros((1,)), 0, 2
        yield np.ones((1,)), 0, 2
        yield np.full((1,), 2), 1, 2
        yield np.full((1,), 3), 1, 2

    return generator


@pytest.fixture
def mock_predict_patches():
    return lambda features, rf, adapter: np.array([[1]] * len(features), dtype=np.uint8)


@pytest.fixture
def mock_get_image_mask():
    return lambda patch_masks, h, w, ps, ov: np.ones((h, w), dtype=np.uint8)


@patch("featureforest.utils.pipeline_prediction.FFImageDataset")
@patch("featureforest.utils.pipeline_prediction.extract_embeddings")
@patch("featureforest.utils.pipeline_prediction.predict_patches")
@patch("featureforest.utils.pipeline_prediction.get_image_mask")
def test_run_prediction_pipeline_yields_correct_masks(
    mock_get_image_mask_func,
    mock_predict_patches_func,
    mock_extract_embeddings_func,
    mock_ffimagedataset,
    mock_model_adapter,
    mock_rf_model,
):
    # Setup mocks
    mock_ffimagedataset.return_value.image_shape = (240, 240)

    # Setup extract_embeddings mock to yield the expected tuples
    def mock_extract_embeddings_generator(*args, **kwargs):
        yield np.zeros((1,)), 0, 2
        yield np.ones((1,)), 0, 2
        yield np.full((1,), 2), 1, 2
        yield np.full((1,), 3), 1, 2

    mock_extract_embeddings_func.side_effect = mock_extract_embeddings_generator

    # Setup other mocks
    mock_predict_patches_func.side_effect = lambda features, rf, adapter: np.array(
        [[1]] * len(features), dtype=np.uint8
    )
    mock_get_image_mask_func.side_effect = lambda patch_masks, h, w, ps, ov: np.ones(
        (h, w), dtype=np.uint8
    )

    input_stack = "dummy_path"
    results = list(
        run_prediction_pipeline(input_stack, mock_model_adapter, mock_rf_model)
    )

    # There should be two yields (one per slice)
    assert len(results) == 2
    for mask, idx, total in results:
        assert mask.shape == (240, 240)
        assert total == 2
        assert idx in (0, 1)
        assert np.all(mask == 1)


@patch("featureforest.utils.pipeline_prediction.FFImageDataset")
@patch("featureforest.utils.pipeline_prediction.extract_embeddings")
@patch("featureforest.utils.pipeline_prediction.predict_patches")
@patch("featureforest.utils.pipeline_prediction.get_image_mask")
def test_run_prediction_pipeline_handles_single_slice(
    mock_get_image_mask_func,
    mock_predict_patches_func,
    mock_extract_embeddings_func,
    mock_ffimagedataset,
    mock_model_adapter,
    mock_rf_model,
):
    # Only one slice
    def single_slice_gen(*args, **kwargs):
        yield np.zeros((1,)), 0, 1
        yield np.ones((1,)), 0, 1

    mock_ffimagedataset.return_value.image_shape = (240, 240)
    mock_extract_embeddings_func.side_effect = single_slice_gen
    mock_predict_patches_func.side_effect = lambda features, rf, adapter: np.array(
        [[1]] * len(features), dtype=np.uint8
    )
    mock_get_image_mask_func.side_effect = lambda patch_masks, h, w, ps, ov: np.ones(
        (h, w), dtype=np.uint8
    )

    input_stack = "dummy_path"
    results = list(
        run_prediction_pipeline(input_stack, mock_model_adapter, mock_rf_model)
    )

    assert len(results) == 1
    mask, idx, total = results[0]
    assert mask.shape == (240, 240)
    assert idx == 0
    assert total == 1
    assert np.all(mask == 1)


@patch("featureforest.utils.pipeline_prediction.FFImageDataset")
@patch("featureforest.utils.pipeline_prediction.extract_embeddings")
@patch("featureforest.utils.pipeline_prediction.predict_patches")
@patch("featureforest.utils.pipeline_prediction.get_image_mask")
def test_run_prediction_pipeline_no_patching_mode(
    mock_get_image_mask_func,
    mock_predict_patches_func,
    mock_extract_embeddings_func,
    mock_ffimagedataset,
    mock_model_adapter,
    mock_rf_model,
):
    """Test that the pipeline works correctly when no_patching is True."""
    # Setup model adapter with no_patching=True
    mock_model_adapter.no_patching = True
    mock_model_adapter.patch_size = 0  # Should be ignored in no_patching mode
    mock_model_adapter.overlap = 0  # Should be ignored in no_patching mode
    mock_model_adapter.get_total_output_channels.return_value = 320

    # Setup dataset
    mock_ffimagedataset.return_value.image_shape = (240, 240)

    # Setup embeddings generator
    def no_patching_gen(*args, **kwargs):
        # In no_patching mode, we'd have one feature per slice
        yield np.zeros((1, 320)), 0, 2
        yield np.ones((1, 320)), 1, 2

    mock_extract_embeddings_func.side_effect = no_patching_gen

    # Setup prediction mocks
    mock_predict_patches_func.side_effect = lambda features, rf, adapter: np.array(
        [[1]] * len(features), dtype=np.uint8
    )
    mock_get_image_mask_func.side_effect = lambda patch_masks, h, w, ps, ov: np.ones(
        (h, w), dtype=np.uint8
    )

    # Run the pipeline
    input_stack = "dummy_path"
    results = list(
        run_prediction_pipeline(input_stack, mock_model_adapter, mock_rf_model)
    )

    # Verify FFImageDataset was created with no_patching=True
    mock_ffimagedataset.assert_called_once_with(
        images="dummy_path", no_patching=True, patch_size=0, overlap=0
    )

    # Verify results
    assert len(results) == 2
    for i, (mask, idx, total) in enumerate(results):
        assert mask.shape == (240, 240)
        assert idx == i
        assert total == 2
        assert np.all(mask == 1)


@patch("featureforest.utils.pipeline_prediction.FFImageDataset")
@patch("featureforest.utils.pipeline_prediction.extract_embeddings")
@patch("featureforest.utils.pipeline_prediction.predict_patches")
@patch("featureforest.utils.pipeline_prediction.get_image_mask")
def test_run_prediction_pipeline_large_image_dimensions(
    mock_get_image_mask_func,
    mock_predict_patches_func,
    mock_extract_embeddings_func,
    mock_ffimagedataset,
    mock_model_adapter,
    mock_rf_model,
):
    """Test that the pipeline handles large image dimensions correctly."""
    # Setup large image dimensions
    large_height, large_width = 4096, 4096
    mock_ffimagedataset.return_value.image_shape = (large_height, large_width)

    # Setup model adapter
    mock_model_adapter.no_patching = False
    mock_model_adapter.patch_size = 256
    mock_model_adapter.overlap = 64
    mock_model_adapter.get_total_output_channels.return_value = 320

    # Setup embeddings generator
    def large_image_gen(*args, **kwargs):
        # Simulate many patches for a large image (simplified for test)
        yield np.zeros((1, 16, 16, 320)), 0, 1
        yield np.ones((1, 16, 16, 320)), 0, 1
        yield np.full((1, 16, 16, 320), 2), 0, 1

    mock_extract_embeddings_func.side_effect = large_image_gen

    # Setup prediction mocks
    mock_predict_patches_func.side_effect = lambda features, rf, adapter: np.array(
        [[1]] * len(features), dtype=np.uint8
    )

    # The get_image_mask should return a mask with the large dimensions
    mock_get_image_mask_func.side_effect = lambda patch_masks, h, w, ps, ov: np.ones(
        (h, w), dtype=np.uint8
    )

    # Run the pipeline
    input_stack = "dummy_path"
    results = list(
        run_prediction_pipeline(input_stack, mock_model_adapter, mock_rf_model)
    )

    # Verify results
    assert len(results) == 1
    mask, idx, total = results[0]
    assert mask.shape == (large_height, large_width)
    assert idx == 0
    assert total == 1
    assert np.all(mask == 1)

    # Verify get_image_mask was called with correct dimensions
    # Instead of checking the exact mock value, just verify the function was called
    # and check that the dimensions and other parameters were correct
    assert mock_get_image_mask_func.called
    args, kwargs = mock_get_image_mask_func.call_args
    # Check that the height, width, patch_size and overlap parameters are correct
    assert args[1] == large_height
    assert args[2] == large_width
    assert args[3] == mock_model_adapter.patch_size
    assert args[4] == mock_model_adapter.overlap


@patch("featureforest.utils.pipeline_prediction.FFImageDataset")
@patch("featureforest.utils.pipeline_prediction.extract_embeddings")
@patch("featureforest.utils.pipeline_prediction.predict_patches")
@patch("featureforest.utils.pipeline_prediction.get_image_mask")
def test_run_prediction_pipeline_slice_index_continuity(
    mock_get_image_mask_func,
    mock_predict_patches_func,
    mock_extract_embeddings_func,
    mock_ffimagedataset,
    mock_model_adapter,
    mock_rf_model,
):
    """Test that the pipeline handles non-continuous slice indices correctly."""
    # Setup
    mock_ffimagedataset.return_value.image_shape = (240, 240)

    # Setup model adapter
    mock_model_adapter.no_patching = False
    mock_model_adapter.patch_size = 128
    mock_model_adapter.overlap = 32
    mock_model_adapter.get_total_output_channels.return_value = 320

    # Setup embeddings generator with non-continuous slice indices (0, 2, 5)
    def non_continuous_slices_gen(*args, **kwargs):
        # Slice 0
        yield np.zeros((1,)), 0, 3
        yield np.ones((1,)), 0, 3
        # Slice 2 (skipping 1)
        yield np.full((1,), 2), 2, 3
        # Slice 5 (skipping 3 and 4)
        yield np.full((1,), 3), 5, 3

    mock_extract_embeddings_func.side_effect = non_continuous_slices_gen

    # Setup prediction mocks
    mock_predict_patches_func.side_effect = lambda features, rf, adapter: np.array(
        [[1]] * len(features), dtype=np.uint8
    )
    mock_get_image_mask_func.side_effect = lambda patch_masks, h, w, ps, ov: np.ones(
        (h, w), dtype=np.uint8
    )

    # Run the pipeline
    input_stack = "dummy_path"
    results = list(
        run_prediction_pipeline(input_stack, mock_model_adapter, mock_rf_model)
    )

    # Verify results
    assert len(results) == 3

    # Check that the slice indices match what we expect
    expected_indices = [0, 2, 5]
    for i, (mask, idx, total) in enumerate(results):
        assert mask.shape == (240, 240)
        assert idx == expected_indices[i]
        assert total == 3
        assert np.all(mask == 1)
