import napari
import napari.utils.notifications as notif
from napari.utils.events import Event
from napari.qt.threading import create_worker
from napari.utils import progress as np_progress

from qtpy.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QCheckBox, QRadioButton,
    QPushButton, QLabel, QComboBox, QLineEdit,
    QFileDialog, QScrollArea, QProgressBar,
    QSizePolicy,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator, QDoubleValidator

import h5py
import numpy as np
import torch
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier

from . import SAM
from .widgets import (
    ScrollWidgetWrapper,
    get_layer, add_labels_layer
)
from .utils import (
    config, colormap
)
from .utils.data import (
    DATA_PATCH_SIZE, TARGET_PATCH_SIZE,
    patchify, unpatchify
)
from .utils.postprocess import (
    process_similarity_matrix, postprocess_label,
    generate_mask_prompts,
)


class SAMRFSegmentationWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_layer = None
        self.gt_layer = None
        self.segmentation_layer = None
        self.storage = None
        self.rf_model = None
        self.device = None
        self.sam_model = None
        self.sam_predictor = None

        self.prepare_widget()

        # init sam model & predictor
        self.sam_model, self.device = self.get_model_on_device()
        self.sam_predictor = SAM.SamPredictor(self.sam_model)

    def prepare_widget(self):
        self.base_layout = QVBoxLayout()
        self.create_input_ui()
        self.create_label_stats_ui()
        self.create_train_ui()
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

        self.viewer.layers.events.inserted.connect(self.check_label_layers)
        self.viewer.layers.events.removed.connect(self.check_label_layers)
        self.check_label_layers(None)

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
        # ground truth layers
        gt_label = QLabel("Ground Truth Layer:")
        self.gt_combo = QComboBox()
        add_labels_button = QPushButton("Add Layer")
        add_labels_button.setMaximumWidth(150)
        add_labels_button.clicked.connect(
            lambda: add_labels_layer(self.viewer)
        )
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
        vbox.addWidget(gt_label)
        hbox = QHBoxLayout()
        hbox.addWidget(self.gt_combo)
        hbox.addWidget(add_labels_button)
        vbox.addLayout(hbox)
        layout.addLayout(vbox)

        gbox = QGroupBox()
        gbox.setTitle("Inputs")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)

    def create_label_stats_ui(self):
        self.num_class_label = QLabel("Number of classes: ")
        self.each_class_label = QLabel("Labels per class:")
        analyze_button = QPushButton("Analyze")
        analyze_button.setMinimumWidth(150)
        analyze_button.setMaximumWidth(150)
        analyze_button.clicked.connect(self.analyze_labels)
        # layout
        layout = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.num_class_label)
        vbox.addWidget(self.each_class_label)
        vbox.addWidget(analyze_button, alignment=Qt.AlignLeft)
        layout.addLayout(vbox)
        gbox = QGroupBox()
        gbox.setTitle("Labeling Statistics")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)

    def create_train_ui(self):
        tree_label = QLabel("Number of trees:")
        self.num_trees_textbox = QLineEdit()
        self.num_trees_textbox.setText("300")
        self.num_trees_textbox.setValidator(QIntValidator(1, 99999))

        depth_label = QLabel("Max depth:")
        self.max_depth_textbox = QLineEdit()
        self.max_depth_textbox.setText("7")
        self.max_depth_textbox.setValidator(QIntValidator(0, 99999))
        self.max_depth_textbox.setToolTip("set to 0 for unlimited depth.")

        train_button = QPushButton("Train RF Model")
        train_button.clicked.connect(self.train_model)
        train_button.setMinimumWidth(150)
        train_button.setMaximumWidth(150)

        self.sam_progress = QProgressBar()
        self.save_storage_button = QPushButton("Save SAM Embeddings")
        # self.save_storage_button.clicked.connect(self.save_embeddings)
        self.save_storage_button.setMinimumWidth(150)
        self.save_storage_button.setMaximumWidth(150)
        self.save_storage_button.setEnabled(False)

        self.model_status_label = QLabel("Model status:")

        load_button = QPushButton("Load Model")
        # load_button.clicked.connect(self.load_model)
        load_button.setMaximumWidth(150)

        self.model_save_button = QPushButton("Save Model")
        # self.model_save_button.clicked.connect(self.save_model)
        self.model_save_button.setMaximumWidth(150)
        self.model_save_button.setEnabled(False)

        # layout
        layout = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(tree_label)
        vbox.addWidget(self.num_trees_textbox)
        vbox.addWidget(depth_label)
        vbox.addWidget(self.max_depth_textbox)
        vbox.addWidget(train_button, alignment=Qt.AlignLeft)
        vbox.addWidget(self.sam_progress)
        vbox.addWidget(self.save_storage_button, alignment=Qt.AlignLeft)
        vbox.addWidget(self.model_status_label)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(load_button)
        hbox.addWidget(self.model_save_button)
        vbox.addLayout(hbox)
        layout.addLayout(vbox)
        gbox = QGroupBox()
        gbox.setTitle("Train Model (Random Forest)")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)

    def create_prediction_ui(self):
        seg_label = QLabel("Segmentation Layer:")
        self.new_layer_checkbox = QCheckBox("New Layer")
        self.new_layer_checkbox.setChecked(True)
        self.new_layer_checkbox.stateChanged.connect(self.new_layer_checkbox_changed)
        self.prediction_layer_combo = QComboBox()
        self.prediction_layer_combo.setEnabled(False)
        self.seg_add_radiobutton = QRadioButton("Add Segmentations")
        self.seg_add_radiobutton.setChecked(True)
        self.seg_add_radiobutton.setEnabled(False)
        self.seg_replace_radionbutton = QRadioButton("Replace Segmentations")
        self.seg_replace_radionbutton.setEnabled(False)

        area_label = QLabel("Area Threshold(%):")
        self.area_threshold_textbox = QLineEdit()
        self.area_threshold_textbox.setText("0.15")
        self.area_threshold_textbox.setValidator(
            QDoubleValidator(0.0, 1.0, 3, notation=QDoubleValidator.StandardNotation)
        )
        self.area_threshold_textbox.setToolTip(
            "Keeps regions with area above the threshold percentage."
        )
        self.area_threshold_textbox.setEnabled(False)

        self.postprocess_checkbox = QCheckBox("Postprocess segmentations")
        self.postprocess_checkbox.stateChanged.connect(
            lambda state: self.area_threshold_textbox.setEnabled(state == Qt.Checked)
        )

        predict_button = QPushButton("Predict")
        # predict_button.clicked.connect(self.predict)
        predict_button.setMinimumWidth(150)
        predict_button.setMaximumWidth(150)

        # layout
        layout = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(seg_label)
        vbox.addWidget(self.new_layer_checkbox)
        vbox.addWidget(self.prediction_layer_combo)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 20)
        hbox.addWidget(self.seg_add_radiobutton)
        hbox.addWidget(self.seg_replace_radionbutton)
        vbox.addLayout(hbox)
        vbox.addWidget(self.postprocess_checkbox)
        vbox.addWidget(area_label)
        vbox.addWidget(self.area_threshold_textbox)
        vbox.addWidget(predict_button, alignment=Qt.AlignLeft)
        layout.addLayout(vbox)
        gbox = QGroupBox()
        gbox.setTitle("Prediction")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)

    def new_layer_checkbox_changed(self):
        state = self.new_layer_checkbox.checkState()
        self.prediction_layer_combo.setEnabled(state == Qt.Unchecked)
        self.seg_add_radiobutton.setEnabled(state == Qt.Unchecked)
        self.seg_replace_radionbutton.setEnabled(state == Qt.Unchecked)

    def check_input_layers(self, event: Event):
        curr_text = self.image_combo.currentText()
        self.image_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.image_combo.addItem(layer.name)
        # put back the selected layer, if not removed
        if len(curr_text) > 0:
            index = self.image_combo.findText(curr_text, Qt.MatchExactly)
            if index > -1:
                self.image_combo.setCurrentIndex(index)

    def check_label_layers(self, event: Event):
        gt_curr_text = self.gt_combo.currentText()
        pred_curr_text = self.prediction_layer_combo.currentText()
        self.gt_combo.clear()
        self.prediction_layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, config.NAPARI_LABELS_LAYER):
                # to handle layer's name change by user
                layer.events.name.disconnect()
                layer.events.name.connect(self.check_label_layers)
                if "Segmentations" in layer.name:
                    self.prediction_layer_combo.addItem(layer.name)
                else:
                    self.gt_combo.addItem(layer.name)
        # put back the selected layers, if not removed
        if len(gt_curr_text) > 0:
            index = self.gt_combo.findText(gt_curr_text, Qt.MatchExactly)
            if index > -1:
                self.gt_combo.setCurrentIndex(index)
        if len(pred_curr_text) > 0:
            index = self.prediction_layer_combo.findText(
                pred_curr_text, Qt.MatchExactly
            )
            if index > -1:
                self.prediction_layer_combo.setCurrentIndex(index)

    def select_storage(self):
        selected_file, _filter = QFileDialog.getOpenFileName(
            self, "Jug Lab", ".", "Embeddings Storage(*.hdf5)"
        )
        if selected_file is not None and len(selected_file) > 0:
            self.storage_textbox.setText(selected_file)
            # load the storage
            self.storage = h5py.File(selected_file, "r")

    def get_class_labels(self):
        labels_dict = {}
        layer = get_layer(
            self.viewer, self.gt_combo.currentText(), config.NAPARI_LABELS_LAYER
        )
        if layer is None:
            print("No label layer is selected!")
            notif.show_error("No label layer is selected!")
            return labels_dict

        class_indices = np.unique(layer.data).tolist()
        # class zero is the napari background class that we should ignore.
        class_indices = [i for i in class_indices if i > 0]
        for class_idx in class_indices:
            positions = np.argwhere(layer.data == class_idx)
            labels_dict[class_idx] = positions

        return labels_dict

    def analyze_labels(self):
        labels_dict = self.get_class_labels()
        num_labels = [len(v) for v in labels_dict.values()]
        self.num_class_label.setText(f"Number of classes: {len(num_labels)}")
        each_class = "\n".join([
            f"class {i + 1}: {num_labels[i]:,d}" for i in range(len(num_labels))
        ])
        self.each_class_label.setText("Labels per class:\n" + each_class)

    def get_model_on_device(self):
        return SAM.setup_lighthq_sam_model()

    def get_train_data(self):
        # get ground truth class labels
        labels_dict = self.get_class_labels()
        if not labels_dict:
            return None
        if self.storage is None:
            notif.show_error("No embeddings storage file is selected!")
            return None

        num_labels = sum([len(v) for v in labels_dict.values()])
        train_data = np.zeros((num_labels, SAM.PATCH_CHANNELS + SAM.EMBEDDING_SIZE))
        labels = np.zeros(num_labels, dtype="int32") - 1
        count = 0
        for class_index in labels_dict:
            for slice_index, y, x in labels_dict[class_index]:
                # slice_features = self.storage[str(slice_index)]["sam"][y, x]
                train_data[count] = self.storage[str(slice_index)]["sam"][y, x]
                labels[count] = class_index
                count += 1
        assert (labels > -1).all()

        return train_data, labels

    def train_model(self):
        self.image_layer = get_layer(
            self.viewer,
            self.image_combo.currentText(), config.NAPARI_IMAGE_LAYER
        )
        if self.image_layer is None:
            notif.show_error("No Image layer is selected!")
            return None

        # get the train data and labels
        dataset = self.get_train_data()
        if dataset is None:
            return
        train_data, labels = dataset

        self.model_status_label.clear()
        self.model_status_label.setText("Model status: Training...")
        notif.show_info("Model status: Training...")
        # train a random forest model
        num_trees = int(self.num_trees_textbox.text())
        max_depth = int(self.max_depth_textbox.text())
        if max_depth == 0:
            max_depth = None
        rf_classifier = RandomForestClassifier(
            n_estimators=num_trees,
            max_depth=max_depth,
            min_samples_leaf=1,
            n_jobs=2,
            verbose=1
        )
        rf_classifier.fit(train_data, labels)
        self.rf_model = rf_classifier
        self.model_status_label.setText("Model status: Ready!")
        notif.show_info("Model status: Training is Done!")
        self.model_save_button.setEnabled(True)
