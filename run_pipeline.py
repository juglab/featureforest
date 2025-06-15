import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pims
import tifffile

from featureforest.models import get_available_models, get_model
from featureforest.postprocess import (
    get_sam_auto_masks,
    postprocess,
    postprocess_with_sam,
    postprocess_with_sam_auto,
)
from featureforest.utils.extract import extract_embeddings_to_file
from featureforest.utils.pipeline_prediction import run_prediction_pipeline


def apply_postprocessing(
    input_image: np.ndarray,
    segmentation_image: np.ndarray,
    smoothing_iterations: int,
    area_threshold: int,
    area_is_absolute: bool,
    use_sam_predictor: bool,
    use_sam_autoseg: bool,
    iou_threshold: float,
) -> dict:
    post_masks = {}
    # simple post-processing
    mask = postprocess(
        segmentation_image, smoothing_iterations, area_threshold, area_is_absolute
    )
    post_masks["post_simple"] = mask

    if use_sam_predictor:
        mask = postprocess_with_sam(
            segmentation_image, smoothing_iterations, area_threshold, area_is_absolute
        )
        post_masks["post_sam"] = mask

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
        post_masks["post_sam_auto"] = mask

    return post_masks


def run(
    input_file: str,
    rf_model_file: str,
    output_dir: str,
    model_name: str = "SAM2_Large",
    no_patching: bool = False,
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
    prediction_dir = segmentation_dir.joinpath("prediction")
    prediction_dir.mkdir(exist_ok=True)
    simple_post_dir = segmentation_dir.joinpath("post_simple")
    simple_post_dir.mkdir(parents=True, exist_ok=True)
    sam_post_dir = segmentation_dir.joinpath("post_sam")
    sam_post_dir.mkdir(parents=True, exist_ok=True)

    # load rf model
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

    # get stack dims
    lazy_stack = pims.open(input_file)
    img_height, img_width = lazy_stack.frame_shape

    # list of available models
    available_models = get_available_models()
    assert model_name in available_models, (
        f"Couldn't find {model_name} in available models\n{available_models}."
    )
    model_adapter = get_model(model_name, img_height, img_width)
    model_adapter.no_patching = no_patching
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    print(f"patch_size: {patch_size}, overlap: {overlap}")

    # post-processing parameters
    do_postprocess = True
    area_is_absolute = True  # is area is based on pixels or percentage (False)
    use_sam_autoseg = False
    sam_autoseg_iou_threshold = 0.35

    # ### Prediction ###
    tic = time.perf_counter()
    for slice_mask, idx, total in run_prediction_pipeline(
        input_stack=input_file,
        model_adapter=model_adapter,
        rf_model=rf_model,
    ):
        print(f"\nslice {idx + 1} / {total}")
        tifffile.imwrite(
            prediction_dir.joinpath(f"slice_{idx:04}_prediction.tiff"), slice_mask
        )

        if do_postprocess:
            print("\nrunning post-processing...")
            slice_img = lazy_stack[idx]
            post_masks = apply_postprocessing(
                slice_img,  # type: ignore
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
                seg_dir = segmentation_dir.joinpath(name)
                # seg_dir.mkdir(exist_ok=True)
                tifffile.imwrite(seg_dir.joinpath(f"slice_{idx:04}_{name}.tiff"), mask)

    print(f"total elapsed time: {(time.perf_counter() - tic)} seconds")


def run_extract_features(
    input_file: str,
    output_dir: str,
    model_name: str = "SAM2_Large",
    no_patching: bool = False,
):
    # input image
    data_path = Path(input_file)
    print(f"data_path exists: {data_path.exists()}")
    # get stack dims
    lazy_stack = pims.open(input_file)
    img_height, img_width = lazy_stack.frame_shape

    # list of available models
    available_models = get_available_models()
    assert model_name in available_models, (
        f"Couldn't find {model_name} in available models\n{available_models}."
    )
    model_adapter = get_model(model_name, img_height, img_width)
    model_adapter.no_patching = no_patching
    patch_size = model_adapter.patch_size
    overlap = model_adapter.overlap
    print(f"patch_size: {patch_size}, overlap: {overlap}")

    # zarr data store
    store_path = Path(output_dir)
    if not store_path.name.endswith(".zarr"):
        output_dir += ".zarr"

    tic = time.perf_counter()
    print(f"extracting features from {input_file}...")
    for idx, total in extract_embeddings_to_file(input_file, output_dir, model_adapter):
        print(f"slice {idx + 1} / {total}")

    print(f"total elapsed time: {(time.perf_counter() - tic)} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="\nFeatureForest run-pipeline script",
    )
    parser.add_argument("--data", help="Path to the input image", required=True)
    parser.add_argument("--outdir", help="Path to the output directory", required=True)
    parser.add_argument("--rf_model", help="Path to the trained RF model", required=False)
    parser.add_argument(
        "--feat_model",
        choices=get_available_models(),
        default="SAM2_Large",
        help="Name of the model for feature extraction",
    )
    parser.add_argument(
        "--no_patching",
        action="store_true",
        help="If true, no patching will be used during feature extraction",
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
        "--post_sam",
        default=True,
        action="store_true",
        help="to use SAM2 for generating final masks",
    )
    parser.add_argument(
        "--only_extract",
        default=False,
        action="store_true",
        help="to only extract features to zarr file without running prediction pipeline",
    )

    args = parser.parse_args()

    if args.only_extract:
        run_extract_features(args.data, args.outdir, args.feat_model, args.no_patching)
        exit(0)

    assert args.rf_model is not None, "RF model file is required."
    run(
        input_file=args.data,
        rf_model_file=args.rf_model,
        output_dir=args.outdir,
        model_name=args.feat_model,
        no_patching=args.no_patching,
        smoothing_iterations=args.smoothing_iterations,
        area_threshold=args.area_threshold,
        use_sam_predictor=args.post_sam,
    )
