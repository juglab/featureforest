from torchvision import transforms

from featureforest.SAM.setup_model import setup_mobile_sam_model
from segment_anything import SamPredictor


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
