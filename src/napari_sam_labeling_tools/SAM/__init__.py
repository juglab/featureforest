from torchvision import transforms

from napari_sam_labeling_tools.SAM.setup_model import setup_lighthq_sam_model
from napari_sam_labeling_tools.SAM.models.segment_anything import SamPredictor


INPUT_SIZE = 1024
FEATURE_H = FEATURE_W = 64
ENCODER_OUT_CHANNELS = 256
PATCH_SIZE = 256
EMBED_PATCH_CHANNELS = 64


sam_transform = transforms.Compose([
    transforms.Resize(
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True
    ),
])
