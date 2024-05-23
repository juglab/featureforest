import napari
import numpy as np

from featureforest.utils import colormaps


def get_layer(napari_viewer, name, layer_types):
    for layer in napari_viewer.layers:
        if layer.name == name and isinstance(layer, layer_types):
            return layer
    return None


def add_labels_layer_(napari_viewer: napari.Viewer):
    """Create new labels layer filling full world coordinates space."""
    # BUGGY
    layers_extent = napari_viewer.layers.extent
    extent = layers_extent.world
    scale = layers_extent.step
    scene_size = extent[1] - extent[0]
    corner = extent[0]
    shape = [
        np.round(s / sc).astype('int') + 1
        for s, sc in zip(scene_size, scale)
    ]
    empty_labels = np.zeros(shape, dtype=np.uint8)
    layer = napari_viewer.add_labels(
        empty_labels, name="Labels", translate=np.array(corner), scale=scale
    )
    layer.colormap = colormaps.create_colormap(10)[0]
