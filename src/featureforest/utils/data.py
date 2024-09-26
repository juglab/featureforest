from typing import Tuple, Optional

import numpy as np
from numpy import ndarray

import torch.nn.functional as F
from torch import Tensor


def get_patch_size(
    img_height: float, img_width: float, divisible_by: float = None
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
    if img_min_dim <= patch_size:
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


def get_stride_margin(patch_size: int, overlap: int) -> Tuple[int, int]:
    """Calculate the patching stride (step), and the margin pad needed to be added to image
        to have complete patches covering all pixels.

    Args:
        patch_size (int): patch size (sliding window size)
        overlap (int): patch overlap size

    Returns:
        Tuple[int, int]: sliding stride and margin
    """
    stride = patch_size - overlap
    margin = overlap // 2
    return stride, margin


def get_paddings(
    patch_size: int, stride: int, margin: int,
    img_height: float, img_width: float
) -> Tuple[int, int]:
    """Calculate the image paddings (right and bottom).

    Args:
        patch_size (int): patch (sliding window) size
        margin (int): margin added to the image before padding
        img_height (int): image height
        img_width (int): image width

    Returns:
        Tuple[int, int]: right and bottom padding
    """
    # pad amount should be enough to make the
    # (final size - patch_size) / stride an integer number.
    # see https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    new_width = img_width + 2 * margin
    pad_right = stride - ((new_width - patch_size) % stride)
    new_height = img_height + 2 * margin
    pad_bottom = stride - ((new_height - patch_size) % stride)

    return pad_right, pad_bottom


def patchify(
    images: Tensor,
    patch_size: Optional[int] = None,
    overlap: Optional[int] = None
) -> Tensor:
    """Divide images into patches.
    images: (B, C, H, W)
    out: (B*N, C, patch_size, patch_size)

    Args:
        images (Tensor): a batch of images of shape (B, C, H, W)
        patch_size (Optional[int], optional): patch size. Defaults to None.
        overlap (Optional[int], optional): patch overlap. Defaults to None.

    Returns:
        Tensor: patches of the input batch of shape (B*N, C, patch_size, patch_size)
    """
    _, c, img_height, img_width = images.shape
    if patch_size is None:
        patch_size = get_patch_size(img_height, img_width)
        overlap = patch_size // 2
    if overlap is None:
        overlap = patch_size // 2

    stride, margin = get_stride_margin(patch_size, overlap)
    pad_right, pad_bottom = get_paddings(patch_size, stride, margin, img_height, img_width)
    pad = (margin, pad_right + margin, margin, pad_bottom + margin)
    padded_imgs = F.pad(images, pad=pad, mode="reflect")
    # making patches using torch unfold method
    patches = padded_imgs.unfold(
        2, patch_size, step=stride).unfold(
        3, patch_size, step=stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        -1, c, patch_size, patch_size
    )

    return patches


def get_num_patches(
    img_height: float, img_width: float, patch_size: int, overlap: int
) -> Tuple[int, int]:
    """Returns number of patches per each image dimension.

    Args:
        img_height (int): image height
        img_width (int): image width
        patch_size (int): patch size
        overlap (int): patch overlap size

    Returns:
        Tuple[int, int]: number of patches for height and width of image
    """
    stride, margin = get_stride_margin(patch_size, overlap)
    pad_right, pad_bottom = get_paddings(patch_size, stride, margin, img_height, img_width)
    new_width = img_width + pad_right + 2 * margin
    num_patches_w = ((new_width - patch_size) / stride) + 1
    assert int(num_patches_w) == num_patches_w, \
        f"number of patches in width {num_patches_w} is not an integer!"
    new_height = img_height + pad_bottom + 2 * margin
    num_patches_h = ((new_height - patch_size) / stride) + 1
    assert int(num_patches_h) == num_patches_h, \
        f"number of patches in height {num_patches_h} is not an integer!"

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

    return patches[
        :, :, margin: margin + stride, margin: margin + stride
    ].permute([0, 2, 3, 1])


def get_patch_index(
    pix_y: float, pix_x: float,
    img_height: float, img_width: float,
    patch_size: int, overlap: int
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
    total_rows, total_cols = get_num_patches(
        img_height, img_width, patch_size, overlap
    )
    stride, _ = get_stride_margin(patch_size, overlap)
    patch_index = (
        pix_y // stride) * total_cols + (pix_x // stride)

    return patch_index


def get_patch_indices(
    pixel_coords: ndarray, img_height: float, img_width: float,
    patch_size: int, overlap: int
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
    total_rows, total_cols = get_num_patches(
        img_height, img_width, patch_size, overlap
    )
    stride, _ = get_stride_margin(patch_size, overlap)
    ys = pixel_coords[:, 0]
    xs = pixel_coords[:, 1]
    patch_indices = (ys // stride) * total_cols + (xs // stride)

    return patch_indices


def get_patch_position(
    pix_y: float, pix_x: float, patch_size: int, overlap: int
) -> Tuple[int, int]:
    """Gets patch 2D position that contains the given pixel coordinate.

    Args:
        pix_y (float): pixel y coordinate
        pix_x (float): pixel x coordinate
        patch_size (int): patch size
        overlap (int): patch overlap size

    Returns:
        Tuple[int, int]: patch position (row, col)
    """
    stride, _ = get_stride_margin(patch_size, overlap)
    patch_row = pix_y // stride
    patch_col = pix_x // stride

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


def get_stack_dims(image_data: ndarray) -> Tuple[int, int, int]:
    """Returns a 3D stack dimensions.

    Args:
        image_data (ndarray): image array

    Returns:
        Tuple[int, int, int]: stack dims; number of slices, height, width
    """
    num_slices = 1
    img_height = image_data.shape[0]
    img_width = image_data.shape[1]
    if is_stacked(image_data):
        num_slices = image_data.shape[0]
        img_height = image_data.shape[1]
        img_width = image_data.shape[2]

    return num_slices, img_height, img_width
