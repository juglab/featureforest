from napari.utils import progress as np_progress

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from featureforest.models import BaseModelAdapter
from featureforest.utils.data import (
    get_stride_margin,
    get_num_patches
)
from featureforest.utils.extract import get_slice_features


def predict_slice(
    rf_model, patch_dataset, model_adapter,
    img_height, img_width, patch_size, overlap
):
    segmentation_image = []
    # shape: N x target_size x target_size x C
    feature_patches = patch_dataset[:]
    num_patches = feature_patches.shape[0]
    total_channels = model_adapter.get_total_output_channels()
    stride, margin = get_stride_margin(patch_size, overlap)

    for i in np_progress(range(num_patches), desc="Predicting slice patches"):
        input_data = feature_patches[i].reshape(-1, total_channels)
        predictions = rf_model.predict(input_data).astype(np.uint8)
        segmentation_image.append(predictions)

    segmentation_image = np.vstack(segmentation_image)
    # reshape into the image size + padding
    patch_rows, patch_cols = get_num_patches(
        img_height, img_width, patch_size, overlap
    )
    segmentation_image = segmentation_image.reshape(
        patch_rows, patch_cols, stride, stride
    )
    segmentation_image = np.moveaxis(segmentation_image, 1, 2).reshape(
        patch_rows * stride,
        patch_cols * stride
    )
    # skip paddings
    segmentation_image = segmentation_image[:img_height, :img_width]

    return segmentation_image


def extract_predict(
        image: np.ndarray,
        model_adapter: BaseModelAdapter,
        storage_group: h5py.Group,
        rf_model: RandomForestClassifier,
        # result_dir: Path
) -> np.ndarray:
    img_height, img_width = image.shape[:2]
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    # remove old dataset from the storage group
    for k in storage_group:
        del storage_group[k]
    # extract the image features into the h5py temp file.
    for b_idx in get_slice_features(
        image, patch_size, overlap, model_adapter, storage_group
    ):
        print(b_idx, end="\r")

    prediction_mask = predict_slice(
        rf_model, storage_group[model_adapter.name], model_adapter,
        img_height, img_width,
        patch_size, overlap
    )

    return prediction_mask
