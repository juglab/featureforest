from typing import Tuple

import torch

from .tiny_vit_sam import TinyViT
from segment_anything.modeling import MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

from featureforest.utils.downloader import download_model
from .adapter import MobileSAMAdapter


def get_model(
    img_height: float, img_width: float, *args, **kwargs
) -> MobileSAMAdapter:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")
    # get the model
    model = setup_model().to(device)
    # download model's weights
    model_url = \
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    model_file = download_model(
        model_url=model_url,
        model_name="mobile_sam.pt"
    )
    if model_file is None:
        raise ValueError(f"Could not download the model from {model_url}.")

    # load weights
    weights = torch.load(model_file, map_location=device)
    model.load_state_dict(weights, strict=True)
    model.eval()

    # create the model adapter
    sam_model_adapter = MobileSAMAdapter(
        model, img_height, img_width, device
    )

    return sam_model_adapter


def setup_model() -> Sam:
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
        image_encoder=TinyViT(
            img_size=1024, in_chans=3, num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    return mobile_sam
