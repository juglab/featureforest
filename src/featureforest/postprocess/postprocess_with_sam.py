from typing import List

from napari.utils import progress as np_progress

import numpy as np
import cv2
import torch

from cv2.typing import Rect
from segment_anything_hq.modeling import Sam
from segment_anything_hq.build_sam import build_sam_vit_t
from segment_anything_hq import SamPredictor

from featureforest.utils.downloader import download_model
from .postprocess import postprocess_label_mask


def get_light_hq_sam() -> Sam:
    """Load the Light HQ SAM model instance. This model produces better masks.

    Raises:
        ValueError: if model's weights could not be downloaded.

    Returns:
        Sam: a Light HQ SAM model instance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # download model's weights
    model_url = \
        "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth"
    model_file = download_model(
        model_url=model_url,
        model_name="sam_hq_vit_tiny.pth"
    )
    if model_file is None:
        raise ValueError(f"Could not download the model from {model_url}.")

    # init & load the light hq sam model
    lhq_sam = build_sam_vit_t().to(device)
    lhq_sam.load_state_dict(
        torch.load(model_file, map_location=device)
    )
    lhq_sam.eval()

    return lhq_sam


def get_watershed_bboxes(image: np.ndarray) -> List[Rect]:
    """Apply watershed algorithm to the input binary image
    to get bounding boxes.

    Args:
        image (np.ndarray): input binary image

    Returns:
        List[Rect]: list of bounding boxes
    """
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


def get_bounding_boxes(image: np.ndarray) -> List[Rect]:
    """Getting bounding boxes around contours in the input image.

    Args:
        image (np.ndarray): input binary image

    Returns:
        List[Rect]: list of bounding boxes
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    for cnt in contours:
        bboxes.append(cv2.boundingRect(cnt))

    return bboxes


def get_sam_mask(
    predictor: SamPredictor, image: np.ndarray, bboxes: List[Rect]
) -> np.ndarray:
    """Returns a mask aggregated by sam predictor masks for each given bounding box.

    Args:
        predictor (SamPredictor): the sam predictor instance
        image (np.ndarray): input binary image
        bboxes (List[Rect]): bounding boxes

    Returns:
        np.ndarray: final mask
    """
    # sam needs an RGB image
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    predictor.set_image(image)
    # get sam-ready bounding boxes: x,y,w,h
    input_boxes = torch.tensor([
        (box[0], box[1], box[0] + box[2], box[1] + box[3])
        for box in bboxes
    ]).to(predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        input_boxes, image.shape[:2]
    )
    # get sam predictor masks
    bs = 16
    num_batches = np.ceil(len(transformed_boxes) / bs).astype(int)
    final_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    for i in np_progress(
        range(num_batches), desc="Generating masks using SAM predictor"
    ):
        start = i * bs
        end = start + bs
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes[start:end],
            multimask_output=True,
        )
        masks_np = masks.squeeze(1).cpu().numpy()
        final_mask = np.bitwise_or(
            final_mask,
            np.bitwise_or.reduce(masks_np, axis=0)
        )

    return final_mask


def postprocess_with_sam(
    segmentation_image: np.ndarray,
    smoothing_iterations: int = 25,
    area_threshold: int = 15,
    area_is_abs: bool = False
) -> np.ndarray:
    """Post-processes segmentations using SAM predictor.

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
    # init a sam predictor using light hq sam
    predictor = SamPredictor(get_light_hq_sam())

    final_image = np.zeros_like(segmentation_image, dtype=np.uint8)
    # postprocessing gets done for each class segmentation.
    bg_label = 0
    class_labels = [c for c in np.unique(segmentation_image) if c > bg_label]
    for label in np_progress(
        class_labels, desc="Getting SAM masks for each class"
    ):
        # make a binary image for the label (class)
        bin_image = (segmentation_image == label).astype(np.uint8) * 255
        processed_mask = postprocess_label_mask(
            bin_image, smoothing_iterations, area_threshold, area_is_abs
        )
        # get component bounding boxes
        w_bboxes = get_watershed_bboxes(processed_mask)
        bboxes = get_bounding_boxes(processed_mask)
        bboxes.extend(w_bboxes)
        # get sam output mask
        final_mask = get_sam_mask(predictor, processed_mask, bboxes)
        # put the final mask into final result image
        final_image[final_mask] = label

    return final_image
