from .base import (
    TiffExporter, NRRDExporter, NumpyExporter
)


EXPORTERS = {
    "tiff": TiffExporter(),
    "nrrd": NRRDExporter(),
    "numpy": NumpyExporter(),
}
