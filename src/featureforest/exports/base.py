# from pathlib import Path

import nrrd
import numpy as np
from tifffile import imwrite
from napari.layers import Layer


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
        tiff_data = layer.data.astype(np.uint8)
        mask_values = np.unique(tiff_data)
        if len(mask_values) == 2:
            # this is a binary mask
            tiff_data[tiff_data == min(mask_values)] = 0
            tiff_data[tiff_data == max(mask_values)] = 255
        imwrite(export_file, tiff_data)


class NRRDExporter(BaseExporter):
    """Export the layer's data into NRRD format."""
    def __init__(self, name: str = "NRRD", extension: str = "nrrd") -> None:
        super().__init__(name, extension)

    def export(self, layer: Layer, export_file: str) -> None:
        nrrd.write(export_file, np.transpose(layer.data))


class NumpyExporter(BaseExporter):
    """Export the layer's data into a numpy array file."""
    def __init__(self, name: str = "Numpy", extension: str = "npy") -> None:
        super().__init__(name, extension)

    def export(self, layer: Layer, export_file: str) -> None:
        return np.save(export_file, layer.data)
