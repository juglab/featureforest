import numpy as np
import cv2
import scipy


# def calc_similarity(vec_a, vec_b):
#     return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def get_furthest_point_from_edge(mask):
    dists = cv2.distanceTransform(
        mask, distanceType=cv2.DIST_L1, maskSize=3
    ).astype(np.float32)
    cy, cx = np.where(dists == dists.max())
    # in sometimes cases multiple values returned for the visual center
    cx, cy = cx.mean(), cy.mean()

    return cx, cy


def process_similarity_matrix(sim_mat):
    """Smooth out given similarity matrix."""
    sim_mat_uint8 = cv2.normalize(sim_mat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    sim_mat_smoothed = cv2.medianBlur(sim_mat_uint8, 13) / 255.

    return sim_mat_smoothed


def postprocess_label(bin_image, area_threshold: float = None):
    # image morphology
    elipse1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    elipse2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed_img = cv2.morphologyEx(bin_image, cv2.MORPH_DILATE, elipse1, iterations=1)
    morphed_img = cv2.morphologyEx(morphed_img, cv2.MORPH_ERODE, elipse2, iterations=1)
    # remove components with small areas
    # stats: left, top, height, width, area
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        morphed_img, connectivity=8, ltype=cv2.CV_32S
    )
    # if there is only background or only one component besides it:
    # num_labels = 1 -> only bg
    # num_labels = 2 -> only one component
    if num_labels < 3:
        return morphed_img

    # get not background(0) areas
    areas = stats[1:, -1]
    if area_threshold is None:
        area_threshold = np.quantile(areas, 0.5)
    else:
        area_threshold = np.quantile(areas, area_threshold)
    small_parts = np.argwhere(stats[:, -1] <= area_threshold)
    morphed_img[np.isin(labels, small_parts)] = 0

    return morphed_img


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
