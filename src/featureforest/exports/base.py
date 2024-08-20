# from pathlib import Path

import nrrd
import numpy as np
from napari.layers import Layer


class BaseExporter:
    def __init__(self, name: str = "Base Exporter", extension: str = "bin") -> None:
        if name.lower() == "tiff":
            raise ValueError(
                "Exporter name is in conflict with default 'tiff' exporter's name."
            )

        self.name = name
        self.extension = extension

    def export(self, layer: Layer, export_file: str) -> None:
        if not export_file.endswith(f".{self.extension}"):
            export_file += f".{self.extension}"
        # implement actual export method here


class TiffExporter(BaseExporter):
    def __init__(self, name: str = "TIFF", extension: str = "tiff") -> None:
        super().__init__(name, extension)

    def export(self, layer: Layer, export_file: str) -> None:
        layer.save(export_file)


class NRRDExporter(BaseExporter):
    def __init__(self, name: str = "NRRD", extension: str = "nrrd") -> None:
        super().__init__(name, extension)

    def export(self, layer: Layer, export_file: str) -> None:
        nrrd.write(export_file, np.transpose(layer.data))


class NumpyExporter(BaseExporter):
    def __init__(self, name: str = "Numpy", extension: str = "npy") -> None:
        super().__init__(name, extension)

    def export(self, layer: Layer, export_file: str) -> None:
        return np.save(export_file, layer.data)
