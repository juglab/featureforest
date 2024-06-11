from tempfile import TemporaryFile

import numpy as np
import pytest
from tqdm import tqdm
import h5py

from featureforest.models import MobileSAM
from featureforest.utils.data import get_stack_dims, get_patch_size
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
def test_embedding_extraction(test_image, expected_shape, expected_slices):
    num_slices, img_height, img_width = get_stack_dims(test_image)
    patch_size = get_patch_size(img_height, img_width)
    overlap = 3 * patch_size // 4

    sam_model, device = MobileSAM.get_model(patch_size, overlap)

    with TemporaryFile() as tmp_file:
        with h5py.File(tmp_file, "w") as write_storage:
            for slice_index in tqdm(range(num_slices)):
                image = test_image[slice_index] if num_slices > 1 else test_image
                slice_grp = write_storage.create_group(str(slice_index))
                get_slice_features(
                    image=image,
                    patch_size=patch_size,
                    overlap=overlap,
                    model_adapter=sam_model.image_encoder,
                    device=device,
                    storage_group=slice_grp,
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
