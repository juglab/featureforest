import numpy as np

from napari.utils import progress as np_progress
from sklearn.ensemble import RandomForestClassifier

from featureforest.utils.data import get_stack_dims, get_patch_indices, get_num_patches
from featureforest.utils.postprocess import postprocess_segmentation
from featureforest.utils.postprocess_with_sam import postprocess_segmentations_with_sam


def create_train_data(
    labels_dict,
    image_layer,
    feature_model,
    patch_size,
    overlap,
    storage,
    stride,
):
    if not labels_dict:
        return None
    num_slices, img_height, img_width = get_stack_dims(image_layer.data)
    num_labels = sum([len(v) for v in labels_dict.values()])
    total_channels = feature_model.get_total_output_channels()
    train_data = np.zeros((num_labels, total_channels))
    labels = np.zeros(num_labels, dtype="int32") - 1
    count = 0
    for class_index in np_progress(
        labels_dict, desc="getting training data", total=len(labels_dict.keys())
    ):
        class_label_coords = labels_dict[class_index]
        uniq_slices = np.unique(class_label_coords[:, 0]).tolist()
        # for each unique slice, load unique patches from the storage,
        # then get the pixel features within loaded patch.
        for slice_index in np_progress(uniq_slices, desc="reading slices"):
            slice_coords = class_label_coords[class_label_coords[:, 0] == slice_index][
                :, 1:
            ]  # omit the slice dim
            patch_indices = get_patch_indices(
                slice_coords, img_height, img_width, patch_size, overlap
            )
            grp_key = str(slice_index)
            slice_dataset = storage[grp_key]["sam"]
            for p_i in np.unique(patch_indices):
                patch_coords = slice_coords[patch_indices == p_i]
                patch_features = slice_dataset[p_i]
                train_data[count : count + len(patch_coords)] = patch_features[
                    patch_coords[:, 0] % stride, patch_coords[:, 1] % stride
                ]
                labels[count : count + len(patch_coords)] = (
                    class_index - 1
                )  # to have bg class as zero
                count += len(patch_coords)
    assert (labels > -1).all()
    return train_data, labels


def train_rf_model(labels, train_data, num_trees, max_depth):
    # train a random forest model
    num_trees = int(num_trees)
    max_depth = int(max_depth)
    if max_depth == 0:
        max_depth = None
    rf_classifier = RandomForestClassifier(
        n_estimators=num_trees,
        max_depth=max_depth,
        min_samples_leaf=1,
        n_jobs=2,
        verbose=1,
    )
    rf_classifier.fit(train_data, labels)
    return rf_classifier


def predict_slice(
    rf_model,
    slice_index,
    img_height,
    img_width,
    postprocessing,
    patch_size,
    overlap,
    storage,
    feature_model,
    stride,
    area_threshold,
    sam_post,
):
    segmentation_image = []
    # shape: N x target_size x target_size x C
    feature_patches = storage[str(slice_index)]["sam"][:]
    num_patches = feature_patches.shape[0]
    total_channels = feature_model.get_total_output_channels()
    for i in np_progress(range(num_patches), desc="Predicting slice patches"):
        input_data = feature_patches[i].reshape(-1, total_channels)
        predictions = rf_model.predict(input_data).astype(np.uint8)
        # to match segmentation colormap with the labels' colormap
        predictions[predictions > 0] += 1
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

    # check for postprocessing
    if postprocessing:
        if sam_post:
            segmentation_image = postprocess_segmentations_with_sam(
                feature_model, segmentation_image, area_threshold
            )
        else:
            segmentation_image = postprocess_segmentation(
                segmentation_image, area_threshold
            )

    return segmentation_image
