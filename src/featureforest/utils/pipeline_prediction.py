from collections.abc import Generator

import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF

from featureforest.models import BaseModelAdapter
from featureforest.utils.data import get_num_patches, get_stride_margin
from featureforest.utils.dataset import FFImageDataset
from featureforest.utils.extract import extract_embeddings


def predict_patches(
    feature_list: list[np.ndarray],
    rf_model: RF,
    model_adapter: BaseModelAdapter,
) -> np.ndarray:
    """Predicts the class labels for a given list of features (patches).

    Args:
        feature_list (list[np.ndarray]): List of features to be predicted.
        rf_model (RF): Random Forest Model used for predictions.
        model_adapter (BaseModelAdapter): Model adapter object used for extracting data.
    """
    patch_masks = []
    # shape: N x stride x stride x C
    patch_features = np.vstack(feature_list)
    num_patches = len(patch_features)
    total_channels = model_adapter.get_total_output_channels()
    print(f"predicting {num_patches} patches...")
    for i in range(num_patches):
        patch_data = patch_features[i].reshape(-1, total_channels)
        pred = rf_model.predict(patch_data).astype(np.uint8)
        patch_masks.append(pred)

    patch_masks = np.vstack(patch_masks)
    return patch_masks


def get_image_mask(
    patch_masks: np.ndarray,
    img_height: int,
    img_width: int,
    patch_size: int,
    overlap: int,
) -> np.ndarray:
    """Gets the final image mask by combining the individual patch masks.

    Args:
        patch_masks (ndarray): Patch masks to combine into an image mask.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.
        patch_size (int): Size of the patches.
        overlap (int): Overlap between adjacent patches.

    Returns:
        np.ndarray: Final image mask.
    """
    stride, _ = get_stride_margin(patch_size, overlap)
    patch_rows, patch_cols = get_num_patches(img_height, img_width, patch_size, overlap)
    mask_image = patch_masks.reshape(patch_rows, patch_cols, stride, stride)
    mask_image = np.moveaxis(mask_image, 1, 2).reshape(
        patch_rows * stride, patch_cols * stride
    )
    # skip paddings
    mask_image = mask_image[:img_height, :img_width]

    return mask_image


def run_prediction_pipeline(
    input_stack: str,
    model_adapter: BaseModelAdapter,
    rf_model: RF,
) -> Generator[tuple[np.ndarray, int, int], None, None]:
    no_patching = model_adapter.no_patching
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    stack_dataset = FFImageDataset(
        images=input_stack,
        no_patching=no_patching,
        patch_size=patch_size,
        overlap=overlap,
    )
    img_height, img_width = stack_dataset.image_shape

    prev_idx = 0
    slice_features = []
    for img_features, slice_idx, total in extract_embeddings(
        model_adapter, image_dataset=stack_dataset
    ):
        print(f"{slice_idx} / {total}")
        if prev_idx != slice_idx:
            # we have one slice features extracted: make a prediction.
            patch_masks = predict_patches(slice_features, rf_model, model_adapter)
            slice_mask = get_image_mask(
                patch_masks, img_height, img_width, patch_size, overlap
            )

            yield slice_mask, prev_idx, total

            # start collecting next slice features
            prev_idx = slice_idx
            slice_features = []
            slice_features.append(img_features)
        else:
            # collect slice features
            slice_features.append(img_features)
    # make prediction for the last slice
    patch_masks = predict_patches(slice_features, rf_model, model_adapter)
    slice_mask = get_image_mask(patch_masks, img_height, img_width, patch_size, overlap)
    yield slice_mask, slice_idx, total
