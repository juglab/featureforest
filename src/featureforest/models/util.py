import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
import imageio.v3 as imageio

from . import _MODELS_DICT, get_model
from ..utils.extract import extract_embeddings_to_file, get_stack_dims


def extract_features(
    image: np.ndarray,
    output_path: Union[os.PathLike, str],
    model_name: str = "SAM2_Tiny",
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> str:
    """Extracts features for the chosen model.

    Args:
        image: The input image.
        output_path: The filepath where the extracted features will be stored.
        model_name: The choice of model that will be used for feature extraction.
        image_height: The height of input image. By default, extracted from the input image.
        image_width: The width of input image. By default, extracted from the input image.

    Returns:
        The filepath where the extracted features have been stored.
    """
    if image_height is None and image_width is None:
        # - Get the height and width of the input image.
        _, image_height, image_width = get_stack_dims(image)

    # Step 1: Get the desired model adapter.
    model_adapter = get_model(model_name=model_name, img_height=image_height, img_width=image_width)

    # - Transform the inputs
    transformed_image = model_adapter.input_transforms(image)

    # Step 2: Run the feature extraction step.
    if os.path.splitext(output_path)[-1].lower() not in [".h5", ".hdf5"]:
        # In this case, we assume that it's a filepath without extension and give it the desired one.
        output_path = str(Path(output_path).with_suffix(".hdf5"))

    extractor_generator = extract_embeddings_to_file(
        image=transformed_image, storage_file_path=output_path, model_adapter=model_adapter,
    )

    # Step 3: Run the extractor generator till the end
    _ = list(extractor_generator)

    return output_path


def main():
    """@private"""

    import argparse

    available_models = list(_MODELS_DICT.keys())
    available_models = ", ".join(available_models)

    parser = argparse.ArgumentParser(description="Extract features for a chosen model.")
    parser.add_argument(
        "-i", "--input_path", type=str, required=True,
        help="The filepath to the image data. Supports all data types that can be read by imageio (eg. tif, png, ...).",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help="The filepath to store the extracted features. The current supports store features in 'h5' / 'hdf5' file.",
    )
    parser.add_argument(
        "--model_choice", type=str, default="SAM2_Tiny",
        help=f"The choice of vision foundation model that will be used, one of ({available_models}). "
        "By default, extracts features for 'SAM2_Tiny'.",
    )

    args = parser.parse_args()

    # Load the image.
    # TODO: Currently supports a simple setup. We can make it more complicated to support other bioformats later.
    image = imageio.imread(args.input_path)

    # Extract the features.
    output_path = extract_features(image=image, output_path=args.output_path, model_name=args.model_choice)

    print(f"The features of '{args.model_choice}' have been extracted at '{os.path.abspath(output_path)}'.")
