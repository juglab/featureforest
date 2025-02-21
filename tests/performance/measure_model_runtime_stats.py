import gc
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
import json
import typer

from featureforest.models import get_model
from featureforest.utils.extract import extract_embeddings_to_file


def measure_runtime_stats(
    image_shape, model_name, storage_path, result_folder="./results"
) -> dict:
    assert len(image_shape) == 2, "Image shape must be a tuple of (height, width)"

    # Create a test image
    test_image = np.random.randint(0, 255, image_shape)
    img_height, img_width = test_image.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hostname = socket.gethostname()

    metrics = {
        "input_shape": (img_height, img_width),
        "device": device,
        "cuda_device_name": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A"
        ),
        "hostname": hostname,
        "model_name": model_name,
    }

    try:
        # Create a model adapter
        model_adapter = get_model(model_name, img_height, img_width, device=device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            initial_memory = 0

        start_time = time.perf_counter()

        for _ in extract_embeddings_to_file(test_image, storage_path, model_adapter):
            pass

        total_wall_time = time.perf_counter() - start_time

        # Get storage file size
        storage_size_mb = (
            os.path.getsize(storage_path) / (1024 * 1024)
            if os.path.exists(storage_path)
            else 0
        )

        metrics.update(
            {
                "total_wall_time": total_wall_time,
                "storage_file_size_mb": storage_size_mb,
            }
        )

        # Aggregate GPU memory metrics
        if torch.cuda.is_available():
            peak_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
            peak_memory_reserved = torch.cuda.max_memory_reserved() / 1024**2
            current_memory_allocated = torch.cuda.memory_allocated() / 1024**2
            current_memory_reserved = torch.cuda.memory_reserved() / 1024**2

            metrics.update(
                {
                    "patch_size": model_adapter.patch_size,
                    "overlap": model_adapter.overlap,
                    "initial_memory": initial_memory,
                    "approximate_max_memory_allocated": initial_memory
                    + peak_memory_reserved,
                    "peak_memory_allocated": peak_memory_allocated,
                    "peak_memory_reserved": peak_memory_reserved,
                    "current_memory_allocated": current_memory_allocated,
                    "current_memory_reserved": current_memory_reserved,
                }
            )

    except RuntimeError as e:
        metrics.update({"failed": True, "error": str(e)})

    if os.path.exists(storage_path):
        os.unlink(storage_path)

    result_file = (
        Path(result_folder) / f"{model_name}_{img_height}_{hostname}_no_ram_track.json"
    )
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(metrics, f)

    return metrics


def main(
    model: str = typer.Argument(..., help="Model name to test"),
    height: int = typer.Argument(..., help="Image height"),
    width: int = typer.Argument(..., help="Image width"),
    storage_path: str = typer.Argument(..., help="Storage path"),
):
    """Measure runtime stats for a specific model and image size."""
    print(f"Testing model: {model} with image shape: ({height}, {width})")
    measure_runtime_stats((height, width), model, storage_path)
    print("Experiment completed")


if __name__ == "__main__":
    typer.run(main)
