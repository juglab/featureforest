from tempfile import TemporaryFile

import numpy as np
import pytest
from tqdm import tqdm
import h5py

from featureforest.SAM import setup_mobile_sam_model
from featureforest.utils.data import get_stack_dims, get_patch_size
from featureforest.utils.extract import get_sam_embeddings_for_slice


@pytest.fixture
def sam_model_and_device():
    sam_model, device = setup_mobile_sam_model()
    return sam_model, device


@pytest.mark.parametrize(
    "test_image, expected_shape, expected_slices",
    [
        (np.ones((256, 256)), (25, 64, 64, 320), 1),  # 2D
        (np.ones((256, 256, 3)), (25, 64, 64, 320), 1),  # 2D RGB
        (np.ones((5, 256, 256)), (25, 64, 64, 320), 5),  # 3D
        (np.ones((5, 256, 256, 3)), (25, 64, 64, 320), 5),  # 3D RGB
    ],
)
def test_embedding_extraction(
    test_image, expected_shape, expected_slices, sam_model_and_device
):
    sam_model, device = sam_model_and_device

    with TemporaryFile() as tmp_file:
        with h5py.File(tmp_file, "w") as write_storage:
            num_slices, img_height, img_width = get_stack_dims(test_image)
            patch_size = get_patch_size(img_height, img_width)
            overlap = 3 * patch_size // 4

            for slice_index in tqdm(range(num_slices)):
                image = test_image[slice_index] if num_slices > 1 else test_image
                slice_grp = write_storage.create_group(str(slice_index))
                get_sam_embeddings_for_slice(
                    image,
                    patch_size,
                    overlap,
                    sam_model.image_encoder,
                    device,
                    slice_grp,
                )

        with h5py.File(tmp_file, "r") as read_storage:
            slices = list(read_storage.keys())
            assert len(slices) == expected_slices, f"Unexpected number of slices"
            for slice in slices:
                slice_key = str(slice)
                slice_dataset = read_storage[slice_key].get("sam")
                assert slice_dataset is not None, f"The dataset is empty"
                assert (
                    slice_dataset.shape == expected_shape
                ), f"Unexpected dataset shape: {slice_dataset.shape}, expected: {expected_shape}"
