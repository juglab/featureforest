# from pathlib import Path

import nrrd
import numpy as np
from tifffile import imwrite
from napari.layers import Layer


def reset_mask_labels(mask_data: np.ndarray) -> np.ndarray:
    """Reset label values in the given mask:
    for a binary mask values will be 0 and 255;
    for a multi-class mask values will be reduced by one to match class index.

    Args:
        mask_data (np.ndarray): input mask

    Returns:
        np.ndarray: fixed mask
    """
    mask_values = np.unique(mask_data)
    if len(mask_values) == 2:
        # this is a binary mask
        mask_data[mask_data == min(mask_values)] = 0
        mask_data[mask_data == max(mask_values)] = 255
    else:
        # reduce one from non-background pixels to match class index
        mask_data[mask_data > 0] -= 1
    assert (mask_data < 0).sum() == 0

    return mask_data

class BaseExporter:
    """Base Exporter Class: all exporters should be a subclass of this class."""
    def __init__(self, name: str = "Base Exporter", extension: str = "bin") -> None:
        self.name = name
        self.extension = extension

    def export(self, layer: Layer, export_file: str) -> None:
        """Export the given layer data

        Args:
            layer (Layer): layer to export the data from
            export_file (str): file path to export
        """
        # implement actual export method here
        return


class TiffExporter(BaseExporter):
    """Export the layer's data into TIFF format."""
    def __init__(self, name: str = "TIFF", extension: str = "tiff") -> None:
        super().__init__(name, extension)

    def export(self, layer: Layer, export_file: str) -> None:
        mask_data = layer.data.copy().astype(np.uint8)
        mask_data = reset_mask_labels(mask_data)
        imwrite(export_file, mask_data)


class NRRDExporter(BaseExporter):
    """Export the layer's data into NRRD format."""
    def __init__(self, name: str = "NRRD", extension: str = "nrrd") -> None:
        super().__init__(name, extension)

    def export(self, layer: Layer, export_file: str) -> None:
        mask_data = layer.data.copy().astype(np.uint8)
        mask_data = reset_mask_labels(mask_data)
        nrrd.write(export_file, np.transpose(mask_data))


class NumpyExporter(BaseExporter):
    """Export the layer's data into a numpy array file."""
    def __init__(self, name: str = "Numpy", extension: str = "npy") -> None:
        super().__init__(name, extension)

    def export(self, layer: Layer, export_file: str) -> None:
        mask_data = layer.data.copy().astype(np.uint8)
        mask_data = reset_mask_labels(mask_data)
        return np.save(export_file, mask_data)
