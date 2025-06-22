from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor


def get_model_ready_image(image: np.ndarray) -> torch.Tensor:
    """Convert the input image to a torch tensor and normalize it.
    Args:
        image (np.ndarray): Input image to be converted (H, W, C).
    Returns:
        torch.Tensor: The input image as a torch tensor, normalized to [0, 1].
    """
    assert image.ndim < 4, "Input image must be 2D or 3D (single channel or RGB)."
    # image to torch tensor
    img_data = torch.from_numpy(image.copy()).to(torch.float32)
    # normalize in [0, 1]
    _min = img_data.min()
    _max = img_data.max()
    img_data = (img_data - _min) / (_max - _min)
    # for image encoders, the input image must be in RGB.
    if is_image_rgb(img_data.numpy()):
        # it's already RGB
        img_data = img_data[..., :3]  # discard the alpha channel (in case of PNG).
        img_data = img_data.permute([2, 0, 1])  # make it channel first
    else:
        # make it RGB by repeating the single channel
        img_data = img_data.unsqueeze(0).expand(3, -1, -1)

    return img_data


def get_patch_size(
    img_height: float, img_width: float, divisible_by: Optional[int] = None
) -> int:
    """Calculate the patch size given image dimensions.

    Args:
        img_height (int): image height
        img_width (int): image width
        divisible_by (float): if given, the patch size must be divisible by it.

    Returns:
        int: patch size
    """
    patch_size = 512
    img_min_dim = min(img_height, img_width)
    # if image is too small
    if img_min_dim < patch_size:
        # get a power of two smaller than image dim
        patch_size = 2 ** int(np.log2(img_min_dim))
    else:
        while img_min_dim / patch_size < 2:
            patch_size = patch_size // 2

    if divisible_by is not None:
        if patch_size < divisible_by:
            raise ValueError("Image dimension is too small!")
        # calc. a patch size which is divisible by the given number
        reminder = patch_size % divisible_by
        if reminder != 0:
            patch_size += divisible_by - reminder

    return patch_size


def get_stride_margin(patch_size: int, overlap: int) -> tuple[int, int]:
    """Calculate the patching stride (step),
        and the margin pad needed to be added to image
        to have complete patches covering all pixels.

    Args:
        patch_size (int): patch size (sliding window size)
        overlap (int): patch overlap size

    Returns:
        tuple[int, int]: sliding stride and margin
    """
    stride = patch_size - overlap
    margin = overlap // 2
    return stride, margin


def get_paddings(
    patch_size: int, stride: int, margin: int, img_height: float, img_width: float
) -> tuple[int, int]:
    """Calculate the image paddings (right and bottom).

    Args:
        patch_size (int): patch (sliding window) size
        margin (int): margin added to the image before padding
        img_height (int): image height
        img_width (int): image width

    Returns:
        tuple[int, int]: right and bottom padding
    """
    # pad amount should be enough to make the
    # (final size - patch_size) / stride an integer number.
    # see https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    # if whole image is just one patch
    if img_height == patch_size and img_width == patch_size and patch_size == stride:
        return 0, 0
    new_width = img_width + 2 * margin
    pad_right = stride - int((new_width - patch_size) % stride)
    new_height = img_height + 2 * margin
    pad_bottom = stride - int((new_height - patch_size) % stride)

    return pad_right, pad_bottom


def patchify(
    image: Tensor, patch_size: Optional[int] = None, overlap: Optional[int] = None
) -> Tensor:
    """Divide images into patches.
    image: (C, H, W)
    out: (N, C, patch_size, patch_size)

    Args:
        images (Tensor): an image of shape (C, H, W)
        patch_size (Optional[int], optional): patch size. Defaults to None.
        overlap (Optional[int], optional): patch overlap. Defaults to None.

    Returns:
        Tensor: patches of shape (N, C, patch_size, patch_size)
    """
    c, img_height, img_width = image.shape
    if patch_size is None:
        patch_size = get_patch_size(img_height, img_width)
        overlap = patch_size // 4
    if overlap is None:
        overlap = patch_size // 4

    stride, margin = get_stride_margin(patch_size, overlap)
    pad_right, pad_bottom = get_paddings(
        patch_size, stride, margin, img_height, img_width
    )
    pad = (margin, pad_right + margin, margin, pad_bottom + margin)
    # add batch dim and pad the image
    padded_imgs = F.pad(image.unsqueeze(0), pad=pad, mode="reflect")
    # making patches using torch unfold method
    patches = padded_imgs.unfold(2, patch_size, step=stride).unfold(
        3, patch_size, step=stride
    )
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, patch_size, patch_size)

    return patches


def get_num_patches(
    img_height: int, img_width: int, patch_size: int, overlap: int
) -> tuple[int, int]:
    """Returns number of patches per each image dimension.

    Args:
        img_height (int): image height
        img_width (int): image width
        patch_size (int): patch size
        overlap (int): patch overlap size

    Returns:
        tuple[int, int]: number of patches for height and width of image
    """
    stride, margin = get_stride_margin(patch_size, overlap)
    pad_right, pad_bottom = get_paddings(
        patch_size, stride, margin, img_height, img_width
    )
    new_width = img_width + pad_right + 2 * margin
    num_patches_w = ((new_width - patch_size) / stride) + 1
    assert int(num_patches_w) == num_patches_w, (
        f"number of patches in width {num_patches_w} is not an integer!"
    )
    new_height = img_height + pad_bottom + 2 * margin
    num_patches_h = ((new_height - patch_size) / stride) + 1
    assert int(num_patches_h) == num_patches_h, (
        f"number of patches in height {num_patches_h} is not an integer!"
    )

    return int(num_patches_h), int(num_patches_w)


def get_nonoverlapped_patches(patches: Tensor, patch_size: int, overlap: int) -> Tensor:
    """Extracts and returns non-overlap patches from patches with overlap.

    Args:
        patches (Tensor): overlapped patches (b,c,h,w)
        patch_size (int): patch size
        overlap (int): patch overlap size

    Returns:
        Tensor: non-overlapped patches (b,h,w,c)
    """
    stride, margin = get_stride_margin(patch_size, overlap)

    return patches[:, :, margin : margin + stride, margin : margin + stride].permute(
        [0, 2, 3, 1]
    )


def get_patch_index(
    pix_y: float,
    pix_x: float,
    img_height: int,
    img_width: int,
    patch_size: int,
    overlap: int,
) -> int:
    """Gets patch index that contains the given single pixel coordinate.

    Args:
        pixel_y (float): pixel y coordinate
        pixel_x (float): pixel x coordinate
        img_height (float): image height
        img_width (float): image width
        patch_size (int): patch size
        overlap (int): patch overlap size

    Returns:
        int: patch index
    """
    total_rows, total_cols = get_num_patches(img_height, img_width, patch_size, overlap)
    stride, _ = get_stride_margin(patch_size, overlap)
    patch_index = int((pix_y // stride) * total_cols + (pix_x // stride))

    return patch_index


def get_patch_indices(
    pixel_coords: ndarray,
    img_height: int,
    img_width: int,
    patch_size: int,
    overlap: int,
) -> ndarray:
    """Gets patch indices that contains the given pixel coordinates.

    Args:
        pixel_coords (ndarray): N x 2 array of pixel coordinates
        img_height (float): image height
        img_width (float): image width
        patch_size (int): patch size
        overlap (int): patch overlap size

    Returns:
        ndarray: array of patch indices
    """
    total_rows, total_cols = get_num_patches(img_height, img_width, patch_size, overlap)
    stride, _ = get_stride_margin(patch_size, overlap)
    ys = pixel_coords[:, 0]
    xs = pixel_coords[:, 1]
    patch_indices = (ys // stride) * total_cols + (xs // stride)

    return patch_indices


def get_patch_position(
    pix_y: float, pix_x: float, patch_size: int, overlap: int
) -> tuple[int, int]:
    """Gets patch 2D position that contains the given pixel coordinate.

    Args:
        pix_y (float): pixel y coordinate
        pix_x (float): pixel x coordinate
        patch_size (int): patch size
        overlap (int): patch overlap size

    Returns:
        tuple[int, int]: patch position (row, col)
    """
    stride, _ = get_stride_margin(patch_size, overlap)
    patch_row = int(pix_y // stride)
    patch_col = int(pix_x // stride)

    return patch_row, patch_col


def is_image_rgb(image_data: ndarray) -> bool:
    """Returns True if the image is an RGB(A) image.

    Args:
        image_data (ndarray): image array

    Returns:
        bool: is image RGB(A)?
    """
    return image_data.shape[-1] in [3, 4]


def is_stacked(image_data: ndarray) -> bool:
    """Returns True if the input is a 3D stack image.

    Args:
        image_data (ndarray): image array

    Returns:
        bool: is image a stack?
    """
    dims = len(image_data.shape)
    if is_image_rgb(image_data):
        return dims == 4
    return dims == 3


def get_stack_dims(image_data: ndarray) -> tuple[int, int, int]:
    """Returns a 3D stack dimensions.

    Args:
        image_data (ndarray): image array

    Returns:
        tuple[int, int, int]: stack dims; number of slices, height, width
    """
    num_slices = 1
    img_height = image_data.shape[0]
    img_width = image_data.shape[1]
    if is_stacked(image_data):
        num_slices = image_data.shape[0]
        img_height = image_data.shape[1]
        img_width = image_data.shape[2]

    return num_slices, img_height, img_width


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalizes an image into [0, 255] as uint8 arrays

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: normalized uint8 image
    """
    _min = image.min()
    _max = image.max()
    if _max - _min == 0:
        _max += 1e-7
    # to prevent invalid value encountered in cast
    normalized_image = np.abs((image - _min) * (255 / (_max - _min))).astype(np.uint8)

    return normalized_image
