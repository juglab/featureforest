import colorsys
import napari
import numpy as np
import matplotlib as mpl


def bit_get(val, idx):
    """Gets the bit value.
    Args:
        val: Input value, int or numpy int array.
        idx: Which bit of the input val.
    Returns:
        The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_colormap2(num_colors=100):
    """Creates a color map for visualizing segmentation results.
    """
    # num_colors += 1  # to omit the first color which is black
    colors = np.zeros((num_colors, 4), dtype=int)
    indices = np.arange(num_colors, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colors[:, channel] |= bit_get(indices, channel) << shift
        indices >>= 3

    # make sure colors' max is 255 (to have brighter colors)
    colors = (
        (colors - colors.min()) * (255 / colors.max() - colors.min())
    ).astype("uint8")  # [1:]  # omit the black color
    # make all colors fully opaque except for the first one (black)
    colors[1:, 3] = 255
    # normalize colors
    colors = colors / 255.
    assert colors.max() == 1

    cm = napari.utils.Colormap(colors, "happy2")

    return cm, colors


def get_colormap1():
    cm = mpl.colormaps.get_cmap("Set3")
    colors = np.array(cm.colors)
    # turn colors into RGBA
    colors = np.hstack((
        colors, np.full((colors.shape[0], 1), 1.0)
    ))
    colors = np.vstack((np.array([[0, 0, 0, 0.]]), colors))
    cm = napari.utils.Colormap(colors, "MySet3")

    return cm


def create_colormap(num_colors, bright=True, black_first=True, seed=4567):
    if num_colors < 10:
        num_colors = 10

    low = [0.0, 0.45, 0.55]
    high = 0.9
    if bright:
        low = [0.0, 0.55, 0.9]
        high = 1.0

    hsv_colors = np.stack([
        np.linspace(start=low[0], stop=high, num=num_colors),
        np.linspace(start=low[1], stop=high, num=num_colors),
        np.linspace(start=low[2], stop=high, num=num_colors)
    ], axis=1)
    rng = np.random.default_rng(seed)
    rng.shuffle(hsv_colors)

    rgba_colors = np.zeros((num_colors, 4))
    for i in range(num_colors):
        rgba_colors[i, :-1] = colorsys.hsv_to_rgb(
            hsv_colors[i, 0], hsv_colors[i, 1], hsv_colors[i, 2]
        )
        rgba_colors[i, -1] = 1

    if black_first:
        rgba_colors = np.vstack((
            np.array([[0, 0, 0, 0]]), rgba_colors
        ))

    cm = napari.utils.Colormap(rgba_colors, "happy")

    return cm, rgba_colors
