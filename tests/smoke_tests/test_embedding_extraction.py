from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pytest

from featureforest.models import get_model
from featureforest.utils.extract import extract_embeddings_to_file
from featureforest.utils.data import get_stack_dims


@pytest.mark.parametrize(
    "test_image, test_model_name, expected_output_shape, expected_slices",
    [
        (np.ones((256, 256)), "MobileSAM", (25, 64, 64, 320), 1),  # 2D
        (np.ones((256, 256, 3)), "MobileSAM", (25, 64, 64, 320), 1),  # 2D RGB
        (np.ones((2, 256, 256)), "MobileSAM", (25, 64, 64, 320), 2),  # 3D
        (np.ones((2, 256, 256, 3)), "MobileSAM", (25, 64, 64, 320), 2),  # 3D RGB
        (np.ones((256, 256)), "SAM", (25, 64, 64, 1536), 1),  # 2D
        (np.ones((256, 256, 3)), "SAM", (25, 64, 64, 1536), 1),  # 2D RGB
        (np.ones((2, 256, 256)), "SAM", (25, 64, 64, 1536), 2),  # 3D
        (np.ones((2, 256, 256, 3)), "SAM", (25, 64, 64, 1536), 2),  # 3D RGB
        (np.ones((256, 256)), "DinoV2", (121, 28, 28, 384), 1),  # 2D
        (np.ones((256, 256, 3)), "DinoV2", (121, 28, 28, 384), 1),  # 2D RGB
        (np.ones((2, 256, 256)), "DinoV2", (121, 28, 28, 384), 2),  # 3D
        (np.ones((2, 256, 256, 3)), "DinoV2", (121, 28, 28, 384), 2),  # 3D RGB
    ],
)
def test_embedding_extraction(
    test_image, test_model_name, expected_output_shape, expected_slices
):
    num_slices, img_height, img_width = get_stack_dims(test_image)
    model_adapter, device = get_model(test_model_name, img_height, img_width)

    with NamedTemporaryFile() as tmp_file:
        extractor_generator = extract_embeddings_to_file(
            image=test_image,
            storage_file_path=tmp_file.name,
            model_adapter=model_adapter,
            device=device,
            model_name=test_model_name,
        )

        # Run the extractor generator till the end
        _ = list(extractor_generator)

        with h5py.File(tmp_file, "r") as read_storage:
            slices = list(read_storage.keys())
            assert (
                len(slices) == expected_slices
            ), f"Unexpected number of slices: {len(slices)}, expected: {expected_slices}"
            for slice in slices:
                slice_key = str(slice)
                slice_dataset = read_storage[slice_key].get(test_model_name)
                assert (
                    slice_dataset is not None
                ), f"The dataset for slice {slice_key} is empty"
                assert (
                    slice_dataset.shape == expected_output_shape
                ), f"Unexpected dataset shape: {slice_dataset.shape}, expected: {expected_output_shape}"
