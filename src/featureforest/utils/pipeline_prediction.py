import multiprocessing as mp

import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF

from featureforest.models import BaseModelAdapter
from featureforest.utils.data import get_num_patches, get_stride_margin
from featureforest.utils.extract import get_slice_features


def predict_patches(
    patch_features: np.ndarray,
    rf_model: RF,
    model_adapter: BaseModelAdapter,
    batch_idx: int,
    result_dict: dict,
) -> None:
    """Predicts the class labels for a given set of patch features.

    Args:
        patch_features (np.ndarray): Patch features to be predicted.
        rf_model (RF): Random Forest Model used for predictions.
        model_adapter (BaseModelAdapter): Model adapter object used for extracting data.
        batch_idx (int): Batch index of the current patch features.
        result_dict (dict): Dictionary where the predicted masks will be stored.
    """
    patch_masks = []
    # shape: N x target_size x target_size x C
    num_patches = patch_features.shape[0]
    total_channels = model_adapter.get_total_output_channels()
    print(f"predicting {num_patches} patches...")
    for i in range(num_patches):
        patch_data = patch_features[i].reshape(-1, total_channels)
        pred = rf_model.predict(patch_data).astype(np.uint8)
        patch_masks.append(pred)

    patch_masks = np.vstack(patch_masks)
    result_dict[batch_idx] = patch_masks


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


def extract_predict(
    image: np.ndarray,
    model_adapter: BaseModelAdapter,
    rf_model: RF,
) -> np.ndarray:
    """Extracts features and predicts the classes for a given image.

    Args:
        image (np.ndarray): Input image to extract features from.
        model_adapter (BaseModelAdapter): Model adapter object used for extracting data.
        rf_model (RF): Random Forest Model used for predictions.

    Returns:
        np.ndarray: Final image mask.
    """
    img_height, img_width = image.shape[:2]
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    procs = []
    # prediction happens per batch of extracted features
    # in a separate process.
    with mp.Manager() as manager:
        result_dict = manager.dict()
        for b_idx, patch_features in get_slice_features(image, model_adapter):
            print(b_idx, end="\r")
            proc = mp.Process(
                target=predict_patches,
                args=(patch_features, rf_model, model_adapter, b_idx, result_dict),
            )
            procs.append(proc)
            proc.start()
        # wait until all processes are done
        for p in procs:
            if p.is_alive:
                p.join()
        # collect results from each process
        batch_indices = sorted(result_dict.keys())
        patch_masks = [result_dict[b] for b in batch_indices]
        patch_masks = np.vstack(patch_masks)
        slice_mask = get_image_mask(
            patch_masks, img_height, img_width, patch_size, overlap
        )

    return slice_mask
