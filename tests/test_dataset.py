import numpy as np
import pytest
import torch

from featureforest.utils.dataset import FFImageDataset


class DummyImageSequence:
    """Dummy class to mock pims.ImageSequence for testing."""

    def __init__(self, images):
        self._images = images
        self.frame_shape = images[0].shape

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        return self._images[idx]

    def __iter__(self):
        return iter(self._images)


@pytest.fixture
def dummy_numpy_stack():
    # shape: (3, 32, 32)
    return np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)


@pytest.fixture
def dummy_numpy_single():
    # shape: (32, 32)
    return np.random.randint(0, 255, (32, 32), dtype=np.uint8)


def test_init_with_numpy_stack(dummy_numpy_stack, monkeypatch):
    ds = FFImageDataset(dummy_numpy_stack, no_patching=True)
    assert ds.num_images == 3
    assert ds.image_shape == (32, 32)
    items = list(ds)
    assert len(items) == 3
    for img, idx in items:
        assert isinstance(img, torch.Tensor)
        assert img.shape[-2:] == (32, 32)
        assert idx[0] in [0, 1, 2]


def test_init_with_numpy_single(dummy_numpy_single):
    ds = FFImageDataset(dummy_numpy_single, no_patching=True)
    assert ds.num_images == 1
    assert ds.image_shape == (32, 32)
    items = list(ds)
    assert len(items) == 1
    img, idx = items[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[-2:] == (32, 32)
    assert idx[0] == 0


def test_iter_with_patching(dummy_numpy_stack, monkeypatch):
    # Patch patchify to split into 2 patches
    monkeypatch.setattr(
        "featureforest.utils.dataset.patchify",
        lambda img, sz, ov: [img[..., :16, :16], img[..., 16:, 16:]],
    )
    ds = FFImageDataset(dummy_numpy_stack, no_patching=False)
    items = list(ds)
    assert len(items) == 6  # 3 images * 2 patches each
    for patch, idx in items:
        assert isinstance(patch, torch.Tensor)
        assert idx.shape == (2,)


def test_init_with_invalid_type():
    with pytest.raises(ValueError):
        FFImageDataset(12345)


def test_init_with_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        FFImageDataset(empty_dir)


def test_init_with_image_files(tmp_path, monkeypatch):
    # Create dummy image files
    for ext in ["tiff", "tif", "png", "jpg"]:
        (tmp_path / f"img1.{ext}").write_bytes(np.random.bytes(100))
    # Patch pims.ImageSequence to DummyImageSequence
    monkeypatch.setattr(
        "featureforest.utils.dataset.pims.ImageSequence",
        lambda files: DummyImageSequence([np.zeros((32, 32), dtype=np.uint8)] * 4),
    )
    ds = FFImageDataset(tmp_path, no_patching=True)
    assert ds.num_images == 4
    assert ds.image_shape == (32, 32)
    items = list(ds)
    assert len(items) == 4


def test_image_shape_none():
    ds = FFImageDataset(np.zeros((1, 2, 2)), no_patching=True)
    ds.image_source = None
    with pytest.raises(ValueError):
        _ = ds.image_shape


def test_iter_no_image_source():
    ds = FFImageDataset(np.zeros((1, 2, 2)), no_patching=True)
    ds.image_source = None
    with pytest.raises(ValueError):
        next(iter(ds))
