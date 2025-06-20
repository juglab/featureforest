from tempfile import TemporaryDirectory

import h5py

from featureforest.utils.extract import extract_embeddings_to_file


def check_embedding_extraction(
    test_image, model_adapter, expected_output_shape, expected_slices
):
    with TemporaryDirectory() as tmp_dir:
        tmp_file = tmp_dir + "/tmp.zarr"

        extractor_generator = extract_embeddings_to_file(
            image=test_image, storage_path=tmp_file, model_adapter=model_adapter
        )

        # Run the extractor generator till the end
        _ = list(extractor_generator)

        read_storage: h5py.File = h5py.File(tmp_file, mode="r")  # type: ignore
        slices = list(read_storage.keys())
        assert len(slices) == expected_slices, (
            f"Unexpected number of slices: {len(slices)}, expected: {expected_slices}"
        )
        for slice_idx in slices:
            slice_key = str(slice_idx)
            slice_dataset = read_storage[slice_key]["features"]
            assert slice_dataset is not None, (
                f"The dataset for slice {slice_key} is empty"
            )
            assert slice_dataset.shape == expected_output_shape, (
                f"Unexpected dataset shape: {slice_dataset.shape}, "
                f"expected: {expected_output_shape}"
            )

        read_storage.close()
