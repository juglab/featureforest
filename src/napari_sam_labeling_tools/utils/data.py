import numpy as np
import torch
import torch.nn.functional as F


# PATCH_EMBEDDING_PATCH_SIZE = 256
IMAGE_PATCH_SIZE = 512
TARGET_PATCH_SIZE = 128


def patchify(imgs, patch_size, target_size):
    """
    imgs: (B, C, H, W)
    out: (B*N, C, patch_size, patch_size)
    """
    b, c, img_h, img_w = imgs.shape
    margin = (patch_size - target_size) // 2
    pad_right = patch_size - (img_w % patch_size) + patch_size - margin
    pad_bottom = patch_size - (img_h % patch_size) + patch_size - margin
    pad = (margin, pad_right - margin, margin, pad_bottom - margin)
    padded_imgs = F.pad(imgs, pad=pad, mode="reflect")
    patches = padded_imgs.unfold(
        2, patch_size, step=target_size).unfold(
        3, patch_size, step=target_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        -1, c, patch_size, patch_size
    )

    return patches


def unpatchify(
    embed_patches, img_h, img_w, patch_size, target_size
):
    """
    patches: (B*N, C, patch_size, patch_size)
    out: (B, embeding_size, H, W)
    """
    bn, c, _, _ = embed_patches.shape
    margin = (patch_size - target_size) // 2
    pad_right = patch_size - (img_w % patch_size) + patch_size - margin
    pad_bottom = patch_size - (img_h % patch_size) + patch_size - margin
    num_patches_w = int((img_w + pad_right - patch_size) / target_size) + 1
    num_patches_h = int((img_h + pad_bottom - patch_size) / target_size) + 1
    total_patches = embed_patches.shape[0]
    batch_size = int(total_patches / (num_patches_h * num_patches_w))
    assert batch_size * num_patches_h * num_patches_w == embed_patches.shape[0]

    batch_patches = embed_patches.reshape(
        batch_size, num_patches_h, num_patches_w, c, patch_size, patch_size
    ).contiguous()
    # print(padded_images.shape)
    target_patches = batch_patches[
        :, :, :, :, margin: margin + target_size, margin: margin + target_size
    ]
    # print(target_patches.shape)  # 1, 26, 22, 3, 16, 16
    target_patches = target_patches.permute(0, 3, 1, 4, 2, 5).reshape(
        batch_size, c, num_patches_h * target_size, num_patches_w * target_size
    )
    # print(target_patches.shape)
    padded_images = torch.zeros((
        batch_size, c, img_h + pad_bottom, img_w + pad_right
    ))
    padded_images[
        :, :, : num_patches_h * target_size, : num_patches_w * target_size
    ] = target_patches

    return padded_images[:, :, :img_h, :img_w]


def unpatchify_np(
    embed_patches, img_h, img_w, patch_size, target_size
):
    """
    patches: (B*N, C, patch_size, patch_size)
    out: (B, embedding_size, H, W)
    """
    bn, c, _, _ = embed_patches.shape
    margin = (patch_size - target_size) // 2
    pad_right = patch_size - (img_w % patch_size) + patch_size - margin
    pad_bottom = patch_size - (img_h % patch_size) + patch_size - margin
    num_patches_w = int((img_w + pad_right - patch_size) / target_size) + 1
    num_patches_h = int((img_h + pad_bottom - patch_size) / target_size) + 1
    total_patches = embed_patches.shape[0]
    batch_size = int(total_patches / (num_patches_h * num_patches_w))
    assert batch_size * num_patches_h * num_patches_w == embed_patches.shape[0]

    batch_patches = embed_patches.reshape(
        batch_size, num_patches_h, num_patches_w, c, patch_size, patch_size
    )
    # print(padded_images.shape)
    target_patches = batch_patches[
        :, :, :, :, margin: margin + target_size, margin: margin + target_size
    ]
    # print(target_patches.shape)
    target_patches = np.transpose(target_patches, [0, 3, 1, 4, 2, 5]).reshape(
        batch_size, c, num_patches_h * target_size, num_patches_w * target_size
    )
    # print(target_patches.shape)
    padded_images = torch.zeros((
        batch_size, c, img_h + pad_bottom, img_w + pad_right
    ))
    padded_images[
        :, :, : num_patches_h * target_size, : num_patches_w * target_size
    ] = target_patches

    return padded_images[:, :, :img_h, :img_w]


def get_num_target_patches(img_height, img_width, patch_size, target_size):
    margin = (patch_size - target_size) // 2
    pad_right = patch_size - (img_width % patch_size) + patch_size - margin
    pad_bottom = patch_size - (img_height % patch_size) + patch_size - margin
    num_patches_w = int((img_width + pad_right - patch_size) / target_size) + 1
    num_patches_h = int((img_height + pad_bottom - patch_size) / target_size) + 1

    return num_patches_h, num_patches_w


def get_target_patches(patches, patch_size, target_size):
    """
    patches: (B, C, patch_size, patch_size)
    out: (
        B, target_size, target_size, C
    )
    """
    margin = (patch_size - target_size) // 2

    return patches[
        :, :, margin: margin + target_size, margin: margin + target_size
    ].permute([0, 2, 3, 1])


def get_patch_index(pixels_y, pixels_x, img_height, img_width, patch_size, target_size):
    """Gets patch index that contains the given pixel coordinates."""
    total_rows, total_cols = get_num_target_patches(
        img_height, img_width, patch_size, target_size
    )
    patch_index = (
        pixels_y // TARGET_PATCH_SIZE) * total_cols + (pixels_x // TARGET_PATCH_SIZE)

    return patch_index


def get_patch_indices(pixel_coords, img_height, img_width, patch_size, target_size):
    """Gets patch index that contains the given pixel coordinates."""
    total_rows, total_cols = get_num_target_patches(
        img_height, img_width, patch_size, target_size
    )
    ys = pixel_coords[:, 0]
    xs = pixel_coords[:, 1]
    patch_indices = (ys // TARGET_PATCH_SIZE) * total_cols + (xs // TARGET_PATCH_SIZE)

    return patch_indices


def get_patch_position(pix_y, pix_x):
    """Gets patch position that contains the given pixel coordinates."""
    # patch_row = int(np.ceil(pix_y / TARGET_PATCH_SIZE))
    # patch_col = int(np.ceil(pix_x / TARGET_PATCH_SIZE))
    patch_row = pix_y // TARGET_PATCH_SIZE
    patch_col = pix_x // TARGET_PATCH_SIZE

    return patch_row, patch_col


def is_image_rgb(image_data):
    return image_data.shape[-1] == 3


def is_stacked(image_data):
    dims = len(image_data.shape)
    if is_image_rgb(image_data):
        return dims == 4
    return dims == 3


def get_stack_sizes(image_data):
    num_slices = 1
    img_height = image_data.shape[0]
    img_width = image_data.shape[1]
    if is_stacked(image_data):
        num_slices = image_data.shape[0]
        img_height = image_data.shape[1]
        img_width = image_data.shape[2]

    return num_slices, img_height, img_width
