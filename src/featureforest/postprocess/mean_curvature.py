import numpy as np
import cv2


def apply_threshold(img, t=None):
    img_copy = img.copy()
    if t is None:
        t, img_copy = cv2.threshold(
            img_copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        img_copy[img_copy > t] = 255
        img_copy[img_copy <= t] = 0

    return img_copy


def get_mean_curvature(input_image: np.ndarray) -> np.ndarray:
    """Returns the mean curvature of the input image.

    Args:
        input_img (np.ndarray): 2D input image

    Returns:
        np.ndarray: smoothed output image (float32)
    """
    # make float copy of the input
    output_image = input_image.copy().astype(np.float32)
    # pad image so the algorithm can be applied to the edges too.
    padded = np.pad(output_image, 1, mode="edge")
    num_rows, num_cols = padded.shape
    # for each column there are right and left columns
    right_indices = np.arange(2, num_cols)
    left_indices = np.arange(0, num_cols - 2)
    # for each row there are top and bottom rows
    top_indices = np.arange(0, num_rows - 2)
    bottom_indices = np.arange(2, num_rows)
    # calculate differences on original image(not padded),
    # so we use (1:-1) to ignore pads.
    # dx: RIGHT(pix) - LEFT(pix)
    dx = padded[1:-1, right_indices] - padded[1:-1, left_indices]
    dx2 = np.power(dx, 2)
    # dy: BOTTOM(pix) - TOP(pix)
    dy = padded[bottom_indices, 1:-1] - padded[top_indices, 1:-1]
    dy2 = np.power(dy, 2)
    # second order differences
    # dxx: RIGHT(pix) + LEFT(pix) - 2 * pix
    dxx = padded[1:-1, right_indices] + padded[1:-1, left_indices] - 2 * output_image
    # dyy: BOTTOM(pix) + TOP(pix) - 2 * pix
    dyy = padded[bottom_indices, 1:-1] + padded[top_indices, 1:-1] - 2 * output_image

    # diagonal neighbors
    dxy = 0.25 * (
        # BOTTOM-RIGHT(pix) - TOP-RIGHT(pix) - BOTTOM-LEFT(pix) + TOP-LEFT(pix)
        padded[2:, 2:] - padded[:-2, 2:] - padded[2:, :-2] + padded[:-2, :-2]
    )
    # mean curvature
    magnitudes = np.sqrt(dx2 + dy2)  # as coefficient of mean curvature
    numerator = dx2 * dyy + dy2 * dxx - 2 * dx * dy * dxy
    denom = np.sqrt(np.power(dx2 + dy2, 3))
    denom[denom == 0] = 1  # to handle zero division
    mean_curvatures = numerator / denom
    output_image += 0.25 * magnitudes * mean_curvatures

    return output_image


def mean_curvature_smoothing(
    input_image: np.ndarray, num_iterations: int = 1
) -> np.ndarray:
    """Smooth the input image by applying mean curvature algorithm in iterations.

    Args:
        input_image (np.ndarray): @d input image
        num_iterations (int, optional): number of smoothing iterations. Defaults to 1.

    Returns:
        np.ndarray: smoothed output image (uint8)
    """
    output_image = get_mean_curvature(input_image)
    for _ in range(num_iterations - 1):
        output_image = get_mean_curvature(output_image)
    # scale image in [0, 255]
    output_image = (output_image - output_image.min()) * (
        255 / (output_image.max() - output_image.min())
    )
    output_image = output_image.astype(np.uint8)
    # apply a threshold to get the final mask
    output_image = apply_threshold(output_image)

    return output_image
