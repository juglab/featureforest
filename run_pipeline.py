import time
import argparse
import multiprocessing as mp
import pickle
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageSequence
from sklearn.ensemble import RandomForestClassifier as RF

from featureforest.models import BaseModelAdapter, get_available_models, get_model
from featureforest.models.SAM import SAMAdapter
from featureforest.postprocess import (
    get_sam_auto_masks,
    postprocess,
    postprocess_with_sam,
    postprocess_with_sam_auto,
)
from featureforest.utils.data import (
    get_num_patches,
    get_stride_margin,
    is_image_rgb,
    patchify,
)


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


def get_slice_features(
    image: np.ndarray, model_adapter: BaseModelAdapter
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Extract features for one image using the given model adapter

    Args:
        image: Input image array
        model_adapter: Model adapter to extract features from
    Returns:
        Generator yielding tuples containing batch index and extracted features.
    """
    # image to torch tensor
    img_data = torch.from_numpy(image).to(torch.float32)
    # normalize in [0, 1]
    _min = img_data.min()
    _max = img_data.max()
    img_data = (img_data - _min) / (_max - _min)
    # for sam the input image should be 4D: BxCxHxW ; an RGB image.
    if is_image_rgb(image):
        # it's already RGB, put the channels first and add a batch dim.
        img_data = img_data[..., :3]  # ignore the Alpha channel (in case of PNG).
        img_data = img_data.permute([2, 0, 1]).unsqueeze(0)
    else:
        img_data = img_data.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

    # get input patches
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    data_patches = patchify(img_data, patch_size, overlap)
    num_patches = len(data_patches)

    # set a low batch size
    batch_size = 8
    # for big SAM we need even lower batch size :(
    if isinstance(model_adapter, SAMAdapter):
        batch_size = 2
    num_batches = int(np.ceil(num_patches / batch_size))

    # get sam encoder output for image patches
    print("extracting slice features:")
    for b_idx in range(num_batches):
        print(f"batch #{b_idx + 1} of {num_batches}")
        start = b_idx * batch_size
        end = start + batch_size
        slice_features = model_adapter.get_features_patches(
            data_patches[start:end].to(model_adapter.device)
        ).cpu()
        if isinstance(slice_features, tuple):  # model with more than one output
            slice_features = torch.cat(slice_features, dim=-1)

        yield b_idx, slice_features.numpy()


def apply_postprocessing(
    input_image: np.ndarray,
    segmentation_image: np.ndarray,
    smoothing_iterations: int,
    area_threshold: int,
    area_is_absolute: bool,
    use_sam_predictor: bool,
    use_sam_autoseg: bool,
    iou_threshold: float,
) -> np.ndarray:
    post_masks = {}
    # if not use_sam_predictor and not use_sam_autoseg:
    mask = postprocess(
        segmentation_image, smoothing_iterations, area_threshold, area_is_absolute
    )
    post_masks["Simple"] = mask

    if use_sam_predictor:
        mask = postprocess_with_sam(
            segmentation_image, smoothing_iterations, area_threshold, area_is_absolute
        )
        post_masks["SAMPredictor"] = mask

    if use_sam_autoseg:
        sam_auto_masks = get_sam_auto_masks(input_image)
        mask = postprocess_with_sam_auto(
            sam_auto_masks,
            segmentation_image,
            smoothing_iterations,
            iou_threshold,
            area_threshold,
            area_is_absolute,
        )
        post_masks["SAMAutoSegmentation"] = mask

    return post_masks


def main(
    input_file: str,
    rf_model_file: str,
    output_dir: str,
    model_name: str = "SAM2_Large",
    smoothing_iterations: int = 25,
    area_threshold: int = 50,
    use_sam_predictor: bool = True,
):
    # input image
    data_path = Path(input_file)
    print(f"data_path exists: {data_path.exists()}")

    # random forest model
    rf_model_path = Path(rf_model_file)
    print(f"rf_model_path exists: {rf_model_path.exists()}")

    # result folder
    segmentation_dir = Path(output_dir)
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = segmentation_dir.joinpath("Prediction")
    prediction_dir.mkdir(exist_ok=True)

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
    assert (
        model_name in available_models
    ), f"Couldn't find {model_name} in available models\n{available_models}."

    model_adapter = get_model(model_name, img_height, img_width)
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    print(f"patch_size: {patch_size}, overlap: {overlap}")

    # post-processing parameters
    do_postprocess = True
    area_is_absolute = True  # is area is based on pixels or percentage (False)
    use_sam_autoseg = False
    sam_autoseg_iou_threshold = 0.35

    # ### Prediction
    tic = time.perf_counter()
    for i, page in enumerate(ImageSequence.Iterator(input_stack)):
        print(f"\nslice {i + 1}")
        slide_img = np.array(page.convert("RGB"))
        procs = []
        # random forest prediction happens per batch of extracted features
        # in a separate process.
        with mp.Manager() as manager:
            result_dict = manager.dict()
            for b_idx, patch_features in get_slice_features(slide_img, model_adapter):
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

        img = Image.fromarray(slice_mask)
        img.save(prediction_dir.joinpath(f"slice_{i:04}_prediction.tiff"))

        if do_postprocess:
            post_masks = apply_postprocessing(
                slide_img,
                slice_mask,
                smoothing_iterations,
                area_threshold,
                area_is_absolute,
                use_sam_predictor,
                use_sam_autoseg,
                sam_autoseg_iou_threshold,
            )
            # save results
            for name, mask in post_masks.items():
                img = Image.fromarray(mask)
                seg_dir = segmentation_dir.joinpath(name)
                seg_dir.mkdir(exist_ok=True)
                img.save(seg_dir.joinpath(f"slice_{i:04}_{name}.tiff"))

    print(f"total elapsed time: {(time.perf_counter() - tic)} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="\nFeatureForest run-pipeline script",
    )
    parser.add_argument("--data", help="Path to the input image", required=True)
    parser.add_argument("--rf_model", help="Path to the trained RF model", required=True)
    parser.add_argument("--outdir", help="Path to the output directory", required=True)
    parser.add_argument(
        "--feat_model",
        choices=get_available_models(),
        help="Name of the model for feature extraction",
    )
    parser.add_argument(
        "--smoothing_iterations",
        default=25,
        type=int,
        help="Post-processing smoothing iterations; default=25",
    )
    parser.add_argument(
        "--area_threshold",
        default=50,
        type=int,
        help="Post-processing area threshold to remove small regions; default=50",
    )
    parser.add_argument(
        "--use_sam_predictor",
        default=True,
        action="store_true",
        help="To use SAM2 for generating final masks",
    )

    args = parser.parse_args()

    main(
        input_file=args.data,
        rf_model_file=args.rf_model,
        output_dir=args.outdir,
        model_name=args.feat_model,
        smoothing_iterations=args.smoothing_iterations,
        area_threshold=args.area_threshold,
        use_sam_predictor=args.use_sam_predictor,
    )
