from tempfile import TemporaryFile

import numpy as np
import pytest
from tqdm import tqdm
import h5py

from featureforest.SAM import setup_mobile_sam_model
from featureforest.utils.data import get_stack_dims
from featureforest.utils.extract import get_sam_embeddings_for_slice


@pytest.fixture
def sam_model_and_device():
    sam_model, device = setup_mobile_sam_model()
    return sam_model, device


@pytest.mark.parametrize(
    "test_image, patch_size, overlap, expected_shape",
    [
        (np.ones((1, 512, 512)), 256, 0, (9, 256, 256, 320)),
        (np.ones((3, 512, 512)), 256, 0, (9, 256, 256, 320)),
        (np.ones((1, 512, 512)), 100, 0, (36, 100, 100, 320)),
        (np.ones((1, 1024, 1024)), 500, 125, (9, 375, 375, 320)),
    ],
)
def test_embedding_extraction(
    test_image, patch_size, overlap, sam_model_and_device, expected_shape
):
    sam_model, device = sam_model_and_device

    with TemporaryFile() as tmp_file:
        with h5py.File(tmp_file, "w") as write_storage:
            num_slices, img_height, img_width = get_stack_dims(test_image)

            for slice_index in tqdm(range(num_slices)):
                image = test_image[slice_index]
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
            groups = list(read_storage.keys())
            for group in groups:
                grp_key = str(group)
                slice_dataset = read_storage[grp_key].get("sam")
                assert slice_dataset is not None
                assert slice_dataset.shape == expected_shape
