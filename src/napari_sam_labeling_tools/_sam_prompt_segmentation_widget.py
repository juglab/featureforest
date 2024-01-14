import napari
import napari.utils.notifications as notif
from napari.utils.events import Event
from napari.qt.threading import create_worker
from napari.utils import progress as np_progress

from qtpy.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QGroupBox, QCheckBox,
    QPushButton, QLabel, QComboBox, QLineEdit,
    QFileDialog, QScrollArea, QProgressBar,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator, QDoubleValidator

import numpy as np
import torch
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier

from . import SAM
from .widgets import (
    ScrollWidgetWrapper,
    get_layer,
)
from .utils import (
    colormaps, config
)
from .utils.data import (
    DATA_PATCH_SIZE, TARGET_PATCH_SIZE,
    patchify, unpatchify
)
from .utils.postprocess import (
    process_similarity_matrix, postprocess_label,
    generate_mask_prompts,
)


class SAMPromptSegmentationWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_layer = None
        self.rf_model = None
        self.device = None
        self.sam_model = None
        self.sam_predictor = None
        self.is_prompt_changed = True
        self.sim_mat = None            # similarity matrix

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
        # sam storage
        storage_label = QLabel("SAM Embeddings Storage:")
        self.storage_textbox = QLineEdit()
        self.storage_textbox.setReadOnly(True)
        storage_button = QPushButton("Select...")
        storage_button.clicked.connect(self.select_storage)
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
        vbox.addWidget(storage_label)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.storage_textbox)
        hbox.addWidget(storage_button)
        vbox.addLayout(hbox)
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
        self.new_layer_checkbox = QCheckBox("Create a new layer")
        self.new_layer_checkbox.setChecked(True)
        self.new_layer_checkbox.stateChanged.connect(
            lambda state: self.prediction_layer_combo.setEnabled(state == Qt.Unchecked)
        )

        lbl = QLabel("Prediction layer:")
        self.prediction_layer_combo = QComboBox()
        self.prediction_layer_combo.setEnabled(False)

        similarity_label = QLabel("Similarity Threshold:")
        self.similarity_threshold_textbox = QLineEdit()
        self.similarity_threshold_textbox.setText("0.91")
        self.similarity_threshold_textbox.setValidator(
            QDoubleValidator(0.000, 1.000, 3, notation=QDoubleValidator.StandardNotation)
        )
        self.similarity_threshold_textbox.setToolTip(
            "Keeps regions having cosine similarity above the threshold with the prompt."
        )

        self.show_intermediate_checkbox = QCheckBox("Show intermediate results")
        self.show_intermediate_checkbox.setChecked(True)

        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self.predict)
        # layout
        layout = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.new_layer_checkbox)
        vbox.addWidget(lbl)
        vbox.addWidget(self.prediction_layer_combo)
        # vbox.addWidget(self.postprocess_checkbox)
        vbox.addWidget(similarity_label)
        vbox.addWidget(self.similarity_threshold_textbox)
        vbox.addWidget(self.show_intermediate_checkbox)
        vbox.addWidget(predict_button)
        layout.addLayout(vbox)
        gbox = QGroupBox()
        gbox.setTitle("Prediction")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)


