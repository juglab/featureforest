import pickle
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageSequence

from featureforest.models import get_available_models, get_model
from featureforest.models.SAM import SAMAdapter
from featureforest.utils.data import (
    patchify,
    is_image_rgb, get_stride_margin,
    get_num_patches
)
from featureforest.postprocess import (
    postprocess,
    postprocess_with_sam, postprocess_with_sam_auto,
    get_sam_auto_masks
)


def get_slice_features(
    image: np.ndarray,
    patch_size: int,
    overlap: int,
    model_adapter,
    storage_group,
):
    """Extract the model features for one slice and save them into storage file."""
    # image to torch tensor
    img_data = torch.from_numpy(image).to(torch.float32) / 255.0
    # for sam the input image should be 4D: BxCxHxW ; an RGB image.
    if is_image_rgb(image):
        # it's already RGB, put the channels first and add a batch dim.
        img_data = img_data[..., :3]  # ignore the Alpha channel (in case of PNG).
        img_data = img_data.permute([2, 0, 1]).unsqueeze(0)
    else:
        img_data = img_data.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

    # get input patches
    data_patches = patchify(img_data, patch_size, overlap)
    num_patches = len(data_patches)

    # set a low batch size
    batch_size = 8
    # for big SAM we need even lower batch size :(
    if isinstance(model_adapter, SAMAdapter):
        batch_size = 2

    num_batches = int(np.ceil(num_patches / batch_size))
    # prepare storage for the slice embeddings
    total_channels = model_adapter.get_total_output_channels()
    stride, _ = get_stride_margin(patch_size, overlap)

    if model_adapter.name not in storage_group:
        dataset = storage_group.create_dataset(
            model_adapter.name, shape=(num_patches, stride, stride, total_channels)
        )
    else:
        dataset = storage_group[model_adapter.name]

    # get sam encoder output for image patches
    print("\nextracting slice features:")
    for b_idx in range(num_batches):
        # print(f"batch #{b_idx + 1} of {num_batches}")
        start = b_idx * batch_size
        end = start + batch_size
        slice_features = model_adapter.get_features_patches(
            data_patches[start:end].to(model_adapter.device)
        )
        if not isinstance(slice_features, tuple):
            # model has only one output
            num_out = slice_features.shape[0]  # to take care of the last batch size
            dataset[start: start + num_out] = slice_features
        else:
            # model has more than one output: put them into storage one by one
            ch_start = 0
            for feat in slice_features:
                num_out = feat.shape[0]
                ch_end = ch_start + feat.shape[-1]  # number of features
                dataset[start: start + num_out, :, :, ch_start:ch_end] = feat
                ch_start = ch_end


def predict_slice(
    rf_model, patch_dataset, model_adapter,
    img_height, img_width, patch_size, overlap
):
    """Predict a slice patch by patch"""
    segmentation_image = []
    # shape: N x target_size x target_size x C
    feature_patches = patch_dataset[:]
    num_patches = feature_patches.shape[0]
    total_channels = model_adapter.get_total_output_channels()
    stride, margin = get_stride_margin(patch_size, overlap)

    print("Predicting slice patches")
    for i in range(num_patches):
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


def apply_postprocessing(
    input_image, segmentation_image,
    smoothing_iterations, area_threshold, area_is_absolute,
    use_sam_predictor, use_sam_autoseg, iou_threshold
):
    post_masks = {}
    # if not use_sam_predictor and not use_sam_autoseg:
    mask = postprocess(
        segmentation_image, smoothing_iterations,
        area_threshold, area_is_absolute
    )
    post_masks["Simple"] = mask

    if use_sam_predictor:
        mask = postprocess_with_sam(
            segmentation_image,
            smoothing_iterations, area_threshold, area_is_absolute
        )
        post_masks["SAMPredictor"] = mask

    if use_sam_autoseg:
        sam_auto_masks = get_sam_auto_masks(input_image)
        mask = postprocess_with_sam_auto(
            sam_auto_masks,
            segmentation_image,
            smoothing_iterations, iou_threshold,
            area_threshold, area_is_absolute
        )
        post_masks["SAMAutoSegmentation"] = mask

    return post_masks


def main(
    data_path, rf_model_path, result_dir,
    model_name="SAM2_Large", storage_path="./temp_storage.hdf5",
    smoothing_iterations=25, area_threshold=50, use_sam_predictor=True
):
    # input image
    data_path = Path(data_path)
    print(f"data_path exists: {data_path.exists()}")

    # random forest model
    rf_model_path = Path(rf_model_path)
    print(f"rf_model_path exists: {rf_model_path.exists()}")

    # result folder
    segmentation_dir = Path(result_dir)
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    # get patch sizes
    input_stack = Image.open(data_path)

    num_slices = input_stack.n_frames
    img_height = input_stack.height
    img_width = input_stack.width

    print(f"input_stack: {num_slices}, {img_height}, {img_width}")

    with open(rf_model_path, mode="rb") as f:
        model_data = pickle.load(f)
    # compatibility check for old format rf model
    if isinstance(model_data, dict):  # noqa: SIM108
        # new format
        rf_model = model_data["rf_model"]
    else:
        # old format
        rf_model = model_data

    rf_model.set_params(verbose=0)
    print(rf_model)

    # list of available models
    available_models = get_available_models()
    assert model_name in available_models, \
        f"Couln't find {model_name} in available models\n{available_models}."

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"running on {device}")
    model_adapter = get_model(model_name, img_height, img_width)
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    print(f"patch_size: {patch_size}, overlap: {overlap}")

    # post-processing parameters
    do_postprocess = True
    area_is_absolute = True      # is area is based on pixels or percentage (False)
    use_sam_autoseg = False
    sam_autoseg_iou_threshold = 0.35

    # ### Prediction
    # create the slice temporary storage
    storage = h5py.File(storage_path, "w")
    storage_group = storage.create_group("slice")

    for i, page in enumerate(ImageSequence.Iterator(input_stack)):
        print(f"\nslice {i + 1}")
        slice_img = np.array(page.convert("RGB"))

        get_slice_features(slice_img, patch_size, overlap, model_adapter, storage_group)

        segmentation_image = predict_slice(
            rf_model, storage_group[model_adapter.name], model_adapter,
            img_height, img_width,
            patch_size, overlap
        )

        img = Image.fromarray(segmentation_image)
        img.save(segmentation_dir.joinpath(f"slice_{i:04}_prediction.tiff"))

        if do_postprocess:
            post_masks = apply_postprocessing(
                slice_img, segmentation_image,
                smoothing_iterations, area_threshold, area_is_absolute,
                use_sam_predictor, use_sam_autoseg, sam_autoseg_iou_threshold
            )
            # save results
            for name, mask in post_masks.items():
                img = Image.fromarray(mask)
                seg_dir = segmentation_dir.joinpath(name)
                seg_dir.mkdir(exist_ok=True)
                img.save(seg_dir.joinpath(f"slice_{i:04}_{name}.tiff"))

    if storage is not None:
        storage.close()
        storage = None
    Path(storage_path).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="\nFeatureForest run-pipeline script",
    )
    parser.add_argument("--data", help="Path to the input image", required=True)
    parser.add_argument("--rf_model", help="Path to the trained RF model", required=True)
    parser.add_argument("--outdir", help="Path to the output directory", required=True)
    parser.add_argument(
        "--feat_model", choices=get_available_models(),
        help="Name of the model for feature extraction"
    )
    parser.add_argument(
        "--storage", default="./temp_storage.hdf5", help="Temporary storage file path"
    )
    parser.add_argument(
        "--smoothing_iterations", default=25, type=int,
        help="Post-processing smoothing iterations; default=25"
    )
    parser.add_argument(
        "--area_threshold", default=50, type=int,
        help="Post-processing area threshold to remove small regions; default=50"
    )
    parser.add_argument(
        "--use_sam_predictor", default=True, action="store_true",
        help="To use SAM2 for generating final masks"
    )

    args = parser.parse_args()

    main(
        data_path=args.data, rf_model_path=args.rf_model, model_name=args.feat_model,
        result_dir=args.outdir, storage_path=args.storage,
        smoothing_iterations=args.smoothing_iterations,
        area_threshold=args.area_threshold,
        use_sam_predictor=args.use_sam_predictor
    )
