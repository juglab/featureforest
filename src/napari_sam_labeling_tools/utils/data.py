import torch
import torch.nn.functional as F


DATA_PATCH_SIZE = 256
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
    # print(f"patches: {patches.shape}")
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
    padded_images = torch.zeros((
        batch_size, c, img_h + pad_bottom, img_w + pad_right
    ))
    # print(padded_images.shape)
    target_patches = batch_patches[
        :, :, :, :, margin: margin + target_size, margin: margin + target_size
    ]
    # print(target_patches.shape)  # 1, 26, 22, 3, 16, 16
    target_patches = target_patches.permute(0, 3, 1, 4, 2, 5).reshape(
        batch_size, c, num_patches_h * target_size, num_patches_w * target_size
    )
    # print(target_patches.shape)
    padded_images[
        :, :, : num_patches_h * target_size, : num_patches_w * target_size
    ] = target_patches

    return padded_images[:, :, :img_h, :img_w]
