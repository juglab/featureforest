from typing import List, Tuple

from napari.utils import progress as np_progress
# import napari.utils.notifications as notif

import numpy as np
import cv2
import torch

from cv2.typing import Rect
from segment_anything_hq.modeling import Sam
from segment_anything_hq.build_sam import build_sam_vit_t
from segment_anything_hq import SamPredictor, SamAutomaticMaskGenerator

from featureforest.utils.downloader import download_model
from featureforest.utils.data import is_image_rgb
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

    final_mask = np.zeros_like(segmentation_image, dtype=np.uint8)
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
        # get bounding boxes around connected components
        w_bboxes = get_watershed_bboxes(processed_mask)
        bboxes = get_bounding_boxes(processed_mask)
        bboxes.extend(w_bboxes)
        # get sam output mask
        sam_label_mask = get_sam_mask(predictor, processed_mask, bboxes)
        # put the final label mask into final result mask
        final_mask[sam_label_mask] = label

    return final_mask


def get_sam_auto_masks(input_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns masks generated by SamAutomaticMaskGenerator

    Args:
        input_image (np.ndarray): input image

    Returns:
        Tuple[np.ndarray, np.ndarray]: a tuple of (masks, areas)
    """
    if not is_image_rgb(input_image):
        input_image = np.repeat(
            input_image[..., np.newaxis],
            3, axis=-1
        )
    assert is_image_rgb(input_image)
    # normalize the image in [0, 255] uint8
    image = input_image.copy()
    _min = image.min()
    _max = image.max()
    image = ((image - _min) * (255 / (_max - _min))).astype(np.uint8)
    # init a sam auto-segmentation mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=get_light_hq_sam(),
        points_per_side=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.85,
        stability_score_offset=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        # crop_nms_thresh=0.7,
        min_mask_region_area=20
    )
    # generate SAM masks
    print("generating masks using SamAutomaticMaskGenerator...")
    with np_progress(range(1), desc="Generating masks using SamAutomaticMaskGenerator"):
        sam_generated_masks = mask_generator.generate(image)
    sam_masks = np.array([mask["segmentation"] for mask in sam_generated_masks])
    sam_areas = np.array([mask["area"] for mask in sam_generated_masks])

    return sam_masks, sam_areas


def get_ious(mask: np.ndarray, sam_masks: np.ndarray) -> np.ndarray:
    """Calculate IOU between prediction mask and all SAM generated masks.

    Args:
        mask (np.ndarray): a single component of the prediction mask (B=1, H, W).
        sam_masks (np.ndarray): SAM generated masks (B, H, W).

    Returns:
        np.ndarray: IOUs array
    """
    epsilon = 1e-6
    intersection = (mask & sam_masks).sum((1, 2))
    union = (mask | sam_masks).sum((1, 2))
    ious = (intersection + epsilon) / (union + epsilon)

    return ious


def postprocess_with_sam_auto(
    sam_auto_masks: Tuple[np.ndarray, np.ndarray],
    segmentation_image: np.ndarray,
    smoothing_iterations: int = 20,
    iou_threshold: float = 0.45,
    area_threshold: int = 15,
    area_is_abs: bool = False
) -> np.ndarray:
    """Post-processes segmentations using SAM auto-segmentation instances' masks.

    Args:
        sam_auto_masks (Tuple[np.ndarray, np.ndarray): a tuple of (masks, areas)
        segmentation_image (np.ndarray): input segmentation image
        smoothing_iterations (int, optional): number of smoothing iterations.
        Defaults to 25.
        iou_threshold (float, optional): IOU threshold for matching prediction masks
        with SAM generated masks. Defaults to 0.45
        area_threshold (int, optional): threshold to remove small regions.
        Defaults to 15.
        area_is_abs (bool, optional): False if the threshold is a percentage.
        Defaults to False.

    Returns:
        np.ndarray: post-processed segmentation image
    """
    sam_masks, sam_areas = sam_auto_masks
    print(f"generated masks: {len(sam_masks)}")

    if iou_threshold > 1.0:
        iou_threshold = 1.0

    final_mask = np.zeros_like(segmentation_image, dtype=np.uint8)
    # postprocessing gets done for each class segmentation.
    bg_label = 0
    class_labels = [c for c in np.unique(segmentation_image) if c > bg_label]
    for class_label in np_progress(
        class_labels, desc="Getting SAM masks for each class"
    ):
        # make a binary image for the label (class)
        bin_image = (segmentation_image == class_label).astype(np.uint8) * 255
        processed_mask = postprocess_label_mask(
            bin_image, smoothing_iterations, area_threshold, area_is_abs
        )
        # get connected components of the label mask
        num_components, component_labels, _, _ = cv2.connectedComponentsWithStats(
            processed_mask, connectivity=8, ltype=cv2.CV_32S
        )
        # component label 0 is bg.
        final_label_mask = np.zeros_like(processed_mask, dtype=bool)
        for cl in np_progress(range(1, num_components), desc="connected component masks"):
            print(f"connected component masks: {cl} / {num_components - 1}", end="\r")
            component_mask = component_labels == cl
            ious = get_ious(component_mask[np.newaxis], sam_masks)
            matched = ious > iou_threshold
            if matched.sum() == 0:
                continue  # no match
            if matched.sum() > 1:
                # multiple match: select one with smallest area
                selected_mask = sam_masks[matched][np.argmin(sam_areas[matched])]
            else:
                selected_mask = sam_masks[matched][0]

            final_label_mask = np.bitwise_or(final_label_mask, selected_mask)
        # put the final label mask into final result mask
        final_mask[final_label_mask.astype(bool)] = class_label

    return final_mask
