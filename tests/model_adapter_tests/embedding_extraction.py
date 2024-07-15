from tempfile import TemporaryDirectory

import h5py

from featureforest.utils.extract import extract_embeddings_to_file


def check_embedding_extraction(
    test_image, model_adapter, expected_output_shape, expected_slices
):
    with TemporaryDirectory() as tmp_dir:
        tmp_file = tmp_dir + "/tmp.h5"

        extractor_generator = extract_embeddings_to_file(
            image=test_image,
            storage_file_path=tmp_file,
            model_adapter=model_adapter
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
                slice_dataset = read_storage[slice_key].get(model_adapter.name)
                assert (
                    slice_dataset is not None
                ), f"The dataset for slice {slice_key} is empty"
                assert (
                    slice_dataset.shape == expected_output_shape
                ), f"Unexpected dataset shape: {slice_dataset.shape}, expected: {expected_output_shape}"