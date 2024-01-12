import numpy as np
import cv2
import torch

from napari_sam_labeling.SAM import SamPredictor


def get_watershed_bboxes(image):
    kernel = np.ones((3, 3), dtype=np.uint8)
    # sure background area
    sure_bg = cv2.dilate(image, kernel, iterations=5)
    # finding sure foreground area
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    # finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # get markers
    ret, markers = cv2.connectedComponents(sure_fg)
    # increase labels by one, so background has 1 as its label
    markers = markers + 1
    # mark the region of unknown with zero
    markers[unknown == 255] = 0
    # watershed
    img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(img_rgb, markers)
    # get watershed contours for each label(component)
    # first two labels are boundaries and background
    watershed_contours = []
    for label in np.unique(markers)[2:]:
        target_img = np.where(markers == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            target_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        watershed_contours.append(contours[0])
    # get bounding boxes
    bboxes = []
    for cnt in watershed_contours:
        bboxes.append(cv2.boundingRect(cnt))

    return bboxes


def get_bounding_boxes(image):
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    for cnt in contours:
        bboxes.append(cv2.boundingRect(cnt))

    return bboxes


def postprocess_label(bin_image, area_threshold: float = None):
    # remove noises
    kernel = np.ones((3, 3), dtype=np.uint8)
    morphed_img = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    morphed_img = cv2.morphologyEx(morphed_img, cv2.MORPH_OPEN, kernel, iterations=2)
    # remove components with small areas
    # stats: left, top, height, width, area
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        morphed_img, connectivity=8, ltype=cv2.CV_32S
    )
    # get not background(0) areas
    areas = stats[1:, -1]
    if area_threshold is None:
        area_threshold = np.quantile(areas, 0.5)
    else:
        area_threshold = np.quantile(areas, area_threshold)
    small_parts = np.argwhere(stats[:, -1] <= area_threshold)
    morphed_img[np.isin(labels, small_parts)] = 0
    print(small_parts.sum(0))

    return morphed_img


def get_sam_mask(predictor, image, bboxes):
    # sam needs an RGB image
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    predictor.set_image(image)
    # get sam-ready bboxes
    input_boxes = torch.tensor([
        (box[0], box[1], box[0] + box[2], box[1] + box[3])
        for box in bboxes
    ]).to(predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        input_boxes, image.shape[:2]
    )
    # get sam predictor masks
    masks, scores, logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
        hq_token_only=False,
    )
    masks_np = masks.squeeze(1).cpu().numpy()
    final_mask = np.bitwise_or.reduce(masks_np, axis=0)

    return final_mask


def postprocess_segmentations_with_sam(
        sam_model, segmentations_image,
        area_threshold: float = None
):
    predictor = SamPredictor(sam_model)
    final_image = np.zeros_like(segmentations_image, dtype=np.uint8)
    # postprocessing gets done for each label's segmentation.
    bg_label = 1
    class_labels = [c for c in np.unique(segmentations_image) if c > bg_label]
    for label in class_labels:
        # make a binary image for the label
        bin_image = (segmentations_image == label).astype(np.uint8) * 255
        processed_label = postprocess_label(bin_image, area_threshold)
        # get component bounding boxes
        w_bboxes = get_watershed_bboxes(processed_label)
        bboxes = get_bounding_boxes(processed_label)
        bboxes.extend(w_bboxes)
        # get sam output mask
        final_mask = get_sam_mask(predictor, processed_label, bboxes)
        # put the processed image into final result image
        final_image[final_mask] = label

    return final_image
