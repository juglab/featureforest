from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pytest
from tqdm import tqdm

from featureforest.models import get_model, get_available_models
from featureforest.utils.data import get_stack_dims
from featureforest.utils.extract import get_slice_features


@pytest.mark.parametrize(
    "test_image, expected_shape, expected_slices",
    [
        (np.ones((256, 256)), (25, 64, 64, 320), 1),  # 2D
        (np.ones((256, 256, 3)), (25, 64, 64, 320), 1),  # 2D RGB
        (np.ones((5, 256, 256)), (25, 64, 64, 320), 5),  # 3D
        (np.ones((5, 256, 256, 3)), (25, 64, 64, 320), 5),  # 3D RGB
    ],
)
@pytest.mark.parametrize("model_name", get_available_models())
def test_embedding_extraction(test_image, expected_shape, expected_slices, model_name):
    num_slices, img_height, img_width = get_stack_dims(test_image)

    model_adapter, device = get_model(model_name, img_height, img_width)

    with NamedTemporaryFile() as tmp_file:
        with h5py.File(tmp_file, "w") as write_storage:
            for slice_index in tqdm(range(num_slices)):
                image = test_image[slice_index] if num_slices > 1 else test_image
                slice_grp = write_storage.create_group(str(slice_index))
                get_slice_features(
                    image=image,
                    patch_size=model_adapter.patch_size,
                    overlap=model_adapter.overlap,
                    model_adapter=model_adapter,
                    device=device,
                    storage_group=slice_grp,
                )

        with h5py.File(tmp_file, "r") as read_storage:
            slices = list(read_storage.keys())
            assert len(slices) == expected_slices, f"Unexpected number of slices: {len(slices)}, expected: {expected_slices}"
            for slice in slices:
                slice_key = str(slice)
                slice_dataset = read_storage[slice_key].get("sam")
                assert slice_dataset is not None, f"The dataset for slice {slice_key} is empty"
                assert (
                    slice_dataset.shape == expected_shape
                ), f"Unexpected dataset shape: {slice_dataset.shape}, expected: {expected_shape}"
