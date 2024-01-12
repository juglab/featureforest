import napari
import napari.utils.notifications as notif
from napari.utils.events import Event

from qtpy.QtWidgets import (
    QVBoxLayout, QWidget,
    QGroupBox,
    QPushButton, QLabel, QComboBox,
)
from qtpy.QtCore import Qt

import numpy as np

from .widgets import (
    ScrollWidgetWrapper,
    get_layer,
)
from . import SAM
from .utils import (
    config, colormap
)


class SAMPredictorWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_layer = None
        self.sam_model = None
        self.device = None
        self.sam_predictor = None
        self.is_prompt_changed = True

        self.prepare_widget()

        # init sam model & predictor
        self.sam_model, self.device = self.get_model_on_device()
        self.sam_predictor = SAM.SamPredictor(self.sam_model)

    def prepare_widget(self):
        self.base_layout = QVBoxLayout()
        self.create_input_ui()
        self.create_prediction_ui()

        scroll_content = QWidget()
        scroll_content.setLayout(self.base_layout)
        scroll = ScrollWidgetWrapper(scroll_content)
        vbox = QVBoxLayout()
        vbox.addWidget(scroll)
        self.setLayout(vbox)
        self.base_layout.addStretch(1)

        self.viewer.layers.events.inserted.connect(self.check_input_layers)
        self.viewer.layers.events.removed.connect(self.check_input_layers)
        self.viewer.layers.events.changed.connect(self.check_input_layers)
        self.check_input_layers(None)

        self.viewer.layers.events.inserted.connect(self.check_prompt_layers)
        self.viewer.layers.events.removed.connect(self.check_prompt_layers)
        self.check_prompt_layers(None)

    def create_input_ui(self):
        # input layer
        input_label = QLabel("Input Layer:")
        self.image_combo = QComboBox()
        # prompt layer
        _label = QLabel("Prompt Layer:")
        self.prompt_combo = QComboBox()
        self.prompt_combo.currentIndexChanged.connect(self.prompt_changed)
        self.prompt_combo.currentTextChanged.connect(self.prompt_changed)
        add_point_prompt_button = QPushButton("Add Point Layer")
        add_point_prompt_button.clicked.connect(lambda: self.add_prompt_layer("point"))
        add_box_prompt_button = QPushButton("Add Box Layer")
        add_box_prompt_button.clicked.connect(lambda: self.add_prompt_layer("box"))

        # layout
        layout = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(input_label)
        vbox.addWidget(self.image_combo)
        layout.addLayout(vbox)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(_label)
        vbox.addWidget(self.prompt_combo)
        vbox.addWidget(add_point_prompt_button)
        vbox.addWidget(add_box_prompt_button)
        layout.addLayout(vbox)

        gbox = QGroupBox()
        gbox.setTitle("Inputs")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)

    def create_prediction_ui(self):
        # predict button
        predict_button = QPushButton("Predict Prompts")
        predict_button.clicked.connect(self.predict_prompts)

        # layout
        layout = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(predict_button)
        layout.addLayout(vbox)

        gbox = QGroupBox()
        gbox.setTitle("Prediction")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)

    def check_input_layers(self, event: Event = None):
        curr_text = self.image_combo.currentText()
        self.image_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, config.NAPARI_IMAGE_LAYER):
                self.image_combo.addItem(layer.name)
        # put back the selected layer, if not removed
        if len(curr_text) > 0:
            index = self.image_combo.findText(curr_text, Qt.MatchExactly)
            if index > -1:
                self.image_combo.setCurrentIndex(index)

    def add_prompt_layer(self, prompt_type: str = "point"):
        layer = None
        if prompt_type == "point":
            layer = self.viewer.add_points(
                name="Point Prompts", ndim=3,
                face_color="lime", edge_color="white", edge_width=0, size=9, opacity=0.85
            )
        else:  # box prompt
            layer = self.viewer.add_shapes(
                name="Box Prompts", ndim=3,
                face_color="#ffffff00", edge_color="lime", edge_width=3, opacity=0.85
            )
        layer.events.data.connect(self.prompt_changed)

    def check_prompt_layers(self, event: Event):
        self.prompt_combo.blockSignals(True)
        curr_prompt_text = self.prompt_combo.currentText()
        self.prompt_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(
                layer, (config.NAPARI_POINTS_LAYER, config.NAPARI_SHAPES_LAYER)
            ):
                # to handle layer's name change by user
                layer.events.name.disconnect()
                layer.events.name.connect(self.check_prompt_layers)
                self.prompt_combo.addItem(layer.name)
        # put back the selected layer, if not removed
        if len(curr_prompt_text) > 0:
            index = self.prompt_combo.findText(curr_prompt_text, Qt.MatchExactly)
            if index > -1:
                self.prompt_combo.setCurrentIndex(index)
            else:
                self.is_prompt_changed = True
        self.prompt_combo.blockSignals(False)

    def prompt_changed(self, event):
        if isinstance(event, Event) and \
                event.source.name == self.prompt_combo.currentText():
            # user added, removed or moved points
            self.is_prompt_changed = True
        else:
            # combobox changed
            self.is_prompt_changed = True

    def get_model_on_device(self):
        return SAM.setup_lighthq_sam_model()

    def get_user_prompts(self):
        user_prompts = None
        if self.prompt_combo.currentIndex() == -1:
            print("No prompt layer is selected!")
            notif.show_error("No prompt layer is selected!")
            return user_prompts
        layer = get_layer(
            self.viewer, self.prompt_combo.currentText(),
            (config.NAPARI_POINTS_LAYER, config.NAPARI_SHAPES_LAYER)
        )
        if layer is None:
            return user_prompts

        user_prompts = layer.data

        return user_prompts

    def get_prompt_labels(self, user_prompts, num_slices, img_height, img_width):
        # point or box prompt
        is_box_prompt = len(user_prompts[0]) == 4
        # make a prediction for each prompt
        prompts_merged_mask = np.zeros(
            (num_slices, img_height, img_width), dtype=np.uint8
        )
        for prompt in user_prompts:
            # first dim of prompt is the slice index.
            # sam prompt need to be as x,y coordinates (numpy is y,x).
            if is_box_prompt:
                slice_index = prompt[0, 0].astype(np.int32)
                input_img = np.repeat(
                    self.image_layer.data[slice_index, :, :, np.newaxis],
                    3, axis=-1
                )
                self.sam_predictor.set_image(input_img)
                # napari box: depends on direction of drawing :( (y, x)
                # SAM box: top-left, bottom-right (x, y)
                top_left = (prompt[:, 2].min(), prompt[:, 1].min())
                bottom_right = (prompt[:, 2].max(), prompt[:, 1].max())
                box = np.array([top_left, bottom_right])
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[np.newaxis, :],
                    multimask_output=True,
                    hq_token_only=False,
                )
            else:
                slice_index = prompt[0].astype(np.int32)
                input_img = np.repeat(
                    self.image_layer.data[slice_index, :, :, np.newaxis],
                    3, axis=-1
                )
                self.sam_predictor.set_image(input_img)
                point = prompt[1:][[1, 0]]
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=point[np.newaxis, :],
                    point_labels=np.array([1]),
                    box=None,
                    multimask_output=True,
                    hq_token_only=False,
                )
            # prompt_mask = masks[0]
            out_mask = np.bitwise_or.reduce(masks, axis=0)
            prompts_merged_mask[slice_index] = prompts_merged_mask[slice_index] | out_mask

        return prompts_merged_mask

    def predict_prompts(self):
        self.image_layer = get_layer(
            self.viewer,
            self.image_combo.currentText(), config.NAPARI_IMAGE_LAYER
        )
        if self.image_layer is None:
            notif.show_error("No Image layer is selected.")
            return None

        user_prompts = self.get_user_prompts()
        if user_prompts is None:
            return

        num_slices, img_height, img_width = self.image_layer.data.shape
        prompts_mask = self.get_prompt_labels(
            user_prompts, num_slices, img_height, img_width
        )
        if prompts_mask.sum() == 0:
            print("SAM model couldn't generate any mask for the given prompts!")
            notif.show_warning(
                "SAM model couldn't generate any mask for the given prompts!"
            )
            return
        # add sam segmentation result as a labels layer
        layer = self.viewer.add_labels(
            data=prompts_mask, name="Prompt Labels",
            opacity=0.6
        )
        layer.colormap = colormap.get_colormap1()
