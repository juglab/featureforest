import napari


def get_layer(napari_viewer, name, layer_types):
    for layer in napari_viewer.layers:
        if layer.name == name and isinstance(layer, layer_types):
            return layer
    return None
