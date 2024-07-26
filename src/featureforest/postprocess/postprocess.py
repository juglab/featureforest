import numpy as np
import cv2

from .mean_curvature import mean_curvature_smoothing


def postprocess_label_mask(
    bin_image: np.ndarray,
    smoothing_iterations: int,
    area_threshold: int,
    area_is_abs: bool
) -> np.ndarray:
    """Post-process a binary mask image (of a class label)

    Args:
        bin_image (np.ndarray): input mask image
        smoothing_iterations (int): number of smoothing iterations
        area_threshold (int): threshold to remove small regions
        area_is_abs (bool): False if the threshold is a percentage

    Returns:
        np.ndarray: post-processed mask image
    """
    # image morphology: trying to close small holes
    elipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed_mask = cv2.morphologyEx(bin_image, cv2.MORPH_DILATE, elipse, iterations=2)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_ERODE, elipse, iterations=2)
    # iterative mean curvature smoothing
    smoothed_mask = mean_curvature_smoothing(smoothed_mask, smoothing_iterations)

    # remove regions with small areas
    # stats: left, top, height, width, area
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        smoothed_mask, connectivity=8, ltype=cv2.CV_32S
    )
    # if there is only background or only one region plus bg:
    # num_labels = 1 -> only bg
    # num_labels = 2 -> only one region
    if num_labels < 3:
        return smoothed_mask

    # get not background areas (not 0)
    areas = stats[1:, -1]
    # if given threshold is a percentage then get corresponding area value
    if not area_is_abs:
        if area_threshold > 100:
            area_threshold = 100
        area_threshold = np.percentile(areas, area_threshold)
    # eliminate small regions
    small_parts = np.argwhere(stats[:, -1] <= area_threshold)
    smoothed_mask[np.isin(labels, small_parts)] = 0

    return smoothed_mask


def postprocess(
    segmentation_image: np.ndarray,
    smoothing_iterations: int = 25,
    area_threshold: int = 15,
    area_is_abs: bool = False
) -> np.ndarray:
    """Post-process a segmentation image mask containing multiple classes.

    Args:
        segmentation_image (np.ndarray): input segmentation image
        smoothing_iterations (int, optional): number of smoothing iterations.
        Defaults to 25.
        area_threshold (int, optional): threshold to remove small regions.
        Defaults to 15.
        area_is_abs (bool, optional): False if the threshold is a percentage.
        Defaults to False.

    Returns:
        np.ndarray: post-processed segmentation image
    """
    final_mask = np.zeros_like(segmentation_image, dtype=np.uint8)
    # postprocessing gets done for each label's segments.
    class_labels = [c for c in np.unique(segmentation_image) if c > 0]
    for label in class_labels:
        # make a binary image for the label
        bin_image = (segmentation_image == label).astype(np.uint8) * 255
        processed_mask = postprocess_label_mask(
            bin_image, smoothing_iterations, area_threshold, area_is_abs
        )
        # put the processed image into final result image
        final_mask[processed_mask == 255] = label

    return final_mask


def process_similarity_matrix(sim_mat):
    """Smooth out given similarity matrix."""
    sim_mat_uint8 = cv2.normalize(sim_mat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    sim_mat_smoothed = cv2.medianBlur(sim_mat_uint8, 13) / 255.

    return sim_mat_smoothed


def get_furthest_point_from_edge(mask):
    dists = cv2.distanceTransform(
        mask, distanceType=cv2.DIST_L1, maskSize=3
    ).astype(np.float32)
    cy, cx = np.where(dists == dists.max())
    # in sometimes cases multiple values returned for the visual center
    cx, cy = cx.mean(), cy.mean()

    return cx, cy


def generate_mask_prompts(mask):
    """
    Generates point prompts out of given mask
    by finding the visual center of contours (approximately).
    # using a fitted ellipse for each contour inside the mask.
    """
    positive_points = []
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        # ellipse = cv2.fitEllipse(cnt)
        # cx = int(ellipse[0][0])
        # cy = int(ellipse[0][1])
        # width, height = ellipse[1]
        # positive_points.append((cx, cy))
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        contour_mask = cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
        cx, cy = get_furthest_point_from_edge(contour_mask)
        positive_points.append((int(cx), int(cy)))
        # add extra points inside the contour
        x2 = cx - 7
        if cv2.pointPolygonTest(cnt, (x2, cy), False) == 1:
            positive_points.append((int(x2), int(cy)))
        y2 = cy + 7
        if cv2.pointPolygonTest(cnt, (cx, y2), False) == 1:
            positive_points.append((int(cx), int(y2)))

    return positive_points
