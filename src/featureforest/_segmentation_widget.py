import os
import pickle
import warnings
from pathlib import Path

import napari
import napari.utils.notifications as notif
from napari.utils.events import Event
from napari.qt.threading import create_worker
from napari.utils import progress as np_progress

from qtpy.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QCheckBox, QRadioButton,
    QPushButton, QLabel, QComboBox, QLineEdit,
    QFileDialog, QProgressBar,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator, QDoubleValidator

import h5py
import numpy as np
import tifffile
from tifffile import TiffFile
from sklearn.ensemble import RandomForestClassifier

from .models import get_model
from .widgets import (
    ScrollWidgetWrapper,
    get_layer,
)
from .utils.data import (
    get_stack_dims, get_patch_indices,
    get_num_patches, get_stride_margin
)
from .utils import (
    colormaps, config
)
from .postprocess import (
    postprocess,
    postprocess_with_sam,
    postprocess_with_sam_auto,
    get_sam_auto_masks
)
from .exports import EXPORTERS
from .utils.pipeline_prediction import (
    extract_predict
)


class SegmentationWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_layer = None
        self.gt_layer = None
        self.segmentation_layer = None
        self.postprocess_layer = None
        self.storage = None
        self.rf_model = None
        self.model_adapter = None
        self.sam_auto_masks = None
        self.patch_size = 512  # default values
        self.overlap = 384
        self.stride = self.patch_size - self.overlap

        self.prepare_widget()

    def closeEvent(self, event):
        print("closing")
        self.viewer.layers.events.inserted.disconnect(self.check_input_layers)
        self.viewer.layers.events.removed.disconnect(self.check_input_layers)
        self.viewer.layers.events.changed.disconnect(self.check_input_layers)

        self.viewer.layers.events.inserted.disconnect(self.check_label_layers)
        self.viewer.layers.events.removed.disconnect(self.check_label_layers)

        self.viewer.layers.events.removed.disconnect(self.postprocess_layer_removed)

    def prepare_widget(self):
        self.base_layout = QVBoxLayout()
        self.create_input_ui()
        self.create_label_stats_ui()
        self.create_train_ui()
        self.create_prediction_ui()
        self.create_postprocessing_ui()
        self.create_export_ui()
        self.create_large_stack_prediction_ui()

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

        self.viewer.layers.events.removed.connect(self.postprocess_layer_removed)

        self.viewer.dims.events.current_step.connect(self.clear_sam_auto_masks)

    def create_input_ui(self):
        # input layer
        input_label = QLabel("Input Layer:")
        self.image_combo = QComboBox()
        self.image_combo.currentIndexChanged.connect(self.clear_sam_auto_masks)
        # sam storage
        storage_label = QLabel("Feature Storage:")
        self.storage_textbox = QLineEdit()
        self.storage_textbox.setReadOnly(True)
        storage_button = QPushButton("Select...")
        storage_button.clicked.connect(self.select_storage)
        # ground truth layers
        gt_label = QLabel("Ground Truth Layer:")
        self.gt_combo = QComboBox()
        add_labels_button = QPushButton("Add Layer")
        add_labels_button.clicked.connect(self.add_labels_layer)
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
        analyze_button.clicked.connect(lambda: self.analyze_labels())
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
        self.num_trees_textbox.setText("450")
        self.num_trees_textbox.setValidator(QIntValidator(1, 99999))

        depth_label = QLabel("Max depth:")
        self.max_depth_textbox = QLineEdit()
        self.max_depth_textbox.setText("9")
        self.max_depth_textbox.setValidator(QIntValidator(0, 99999))
        self.max_depth_textbox.setToolTip("set to 0 for unlimited depth.")

        train_button = QPushButton("Train RF Model")
        train_button.clicked.connect(self.train_model)
        train_button.setMinimumWidth(150)

        self.model_status_label = QLabel("Model status:")

        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_rf_model)
        load_button.setMinimumWidth(150)

        self.model_save_button = QPushButton("Save Model")
        self.model_save_button.clicked.connect(self.save_rf_model)
        self.model_save_button.setMinimumWidth(150)
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
        # vbox.addWidget(self.sam_progress)
        # vbox.addWidget(self.save_storage_button, alignment=Qt.AlignLeft)
        vbox.addWidget(self.model_status_label)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(load_button, alignment=Qt.AlignLeft)
        hbox.addWidget(self.model_save_button, alignment=Qt.AlignLeft)
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
        self.seg_add_radiobutton.setChecked(False)
        self.seg_add_radiobutton.setEnabled(False)
        self.seg_replace_radiobutton = QRadioButton("Replace Segmentations")
        self.seg_replace_radiobutton.setChecked(True)
        self.seg_replace_radiobutton.setEnabled(False)

        predict_button = QPushButton("Predict Slice")
        predict_button.setMinimumWidth(150)
        predict_button.clicked.connect(lambda: self.predict(whole_stack=False))
        self.predict_all_button = QPushButton("Predict Whole Stack")
        self.predict_all_button.setMinimumWidth(150)
        self.predict_all_button.clicked.connect(lambda: self.predict(whole_stack=True))
        self.predict_stop_button = QPushButton("Stop")
        self.predict_stop_button.clicked.connect(self.stop_predict)
        self.predict_stop_button.setMinimumWidth(150)
        self.predict_stop_button.setEnabled(False)
        self.prediction_progress = QProgressBar()

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
        hbox.addWidget(self.seg_replace_radiobutton)
        vbox.addLayout(hbox)
        vbox.addWidget(predict_button, alignment=Qt.AlignLeft)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.predict_all_button, alignment=Qt.AlignLeft)
        hbox.addWidget(self.predict_stop_button, alignment=Qt.AlignLeft)
        vbox.addLayout(hbox)
        vbox.addWidget(self.prediction_progress)
        layout.addLayout(vbox)

        gbox = QGroupBox()
        gbox.setTitle("Prediction")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)

    def create_postprocessing_ui(self):
        smooth_label = QLabel("Smoothing Iterations:")
        self.smoothing_iteration_textbox = QLineEdit()
        self.smoothing_iteration_textbox.setText("25")
        self.smoothing_iteration_textbox.setValidator(
            QIntValidator(0, 1000000)
        )

        area_label = QLabel("Area Threshold:")
        self.area_threshold_textbox = QLineEdit()
        self.area_threshold_textbox.setText("50")
        self.area_threshold_textbox.setValidator(
            QIntValidator(0, 2147483647)
        )
        self.area_threshold_textbox.setToolTip(
            "Keeps only regions with area above the threshold."
        )
        self.area_abs_radiobutton = QRadioButton("absolute")
        self.area_abs_radiobutton.setChecked(True)
        self.area_percent_radiobutton = QRadioButton("percentage")

        self.sam_post_checkbox = QCheckBox("Use SAM Predictor")
        self.sam_post_checkbox.clicked.connect(self.sam_post_checked)
        sam_label = QLabel(
            "This will generate prompts for SAM Predictor using bounding boxes"
            " around segmented regions."
        )
        sam_label.setWordWrap(True)

        self.sam_auto_post_checkbox = QCheckBox("Use SAM Auto-Segmentation")
        self.sam_auto_post_checkbox.clicked.connect(self.sam_auto_post_checked)
        sam_threshold_label = QLabel("IOU Matching Threshold:")
        self.sam_auto_threshold_textbox = QLineEdit()
        validator = QDoubleValidator(0.0, 1.0, 2)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.sam_auto_threshold_textbox.setValidator(validator)
        self.sam_auto_threshold_textbox.setText("0.35")
        self.sam_auto_threshold_textbox.setToolTip(
            "Keeps prediction mask regions with having IOU against SAM generated masks"
            " above the threshold (should be between [0, 1])."
        )
        sam_auto_label = QLabel(
            "This will use SAM auto-segmentation instances' masks to generate"
            " final semantic segmentation mask."
        )
        sam_auto_label.setWordWrap(True)

        postprocess_button = QPushButton("Apply to Slice")
        postprocess_button.setMinimumWidth(150)
        postprocess_button.clicked.connect(self.postprocess_segmentation)
        postprocess_all_button = QPushButton("Apply to Stack")
        postprocess_all_button.setMinimumWidth(150)
        postprocess_all_button.clicked.connect(
            lambda: self.postprocess_segmentation(whole_stack=True)
        )

        layout = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(smooth_label)
        vbox.addWidget(self.smoothing_iteration_textbox)
        vbox.addWidget(area_label)
        hbox = QHBoxLayout()
        hbox.addWidget(self.area_abs_radiobutton)
        hbox.addWidget(self.area_percent_radiobutton)
        vbox.addLayout(hbox)
        vbox.addWidget(self.area_threshold_textbox)
        vbox.addSpacing(15)
        vbox.addWidget(self.sam_post_checkbox)
        vbox.addWidget(sam_label)
        vbox.addSpacing(3)
        vbox.addWidget(self.sam_auto_post_checkbox)
        vbox.addWidget(sam_threshold_label)
        vbox.addWidget(self.sam_auto_threshold_textbox)
        vbox.addWidget(sam_auto_label)
        vbox.addSpacing(7)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(postprocess_button, alignment=Qt.AlignLeft)
        hbox.addWidget(postprocess_all_button, alignment=Qt.AlignLeft)
        vbox.addLayout(hbox)
        # vbox.addSpacing(20)
        layout.addLayout(vbox)

        gbox = QGroupBox()
        gbox.setTitle("Post-processing")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)

    def create_export_ui(self):
        export_label = QLabel("Export Format:")
        self.export_format_combo = QComboBox()
        for exporter in EXPORTERS:
            self.export_format_combo.addItem(exporter)

        self.export_postprocess_checkbox = QCheckBox("Export with Post-processing")
        self.export_postprocess_checkbox.setChecked(True)
        self.export_postprocess_checkbox.setToolTip(
            "Export segmentation result with applied post-processing, if available."
        )

        export_button = QPushButton("Export")
        export_button.setMinimumWidth(150)
        export_button.clicked.connect(self.export_segmentation)

        # layout
        layout = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(export_label)
        hbox.addWidget(self.export_format_combo)
        vbox.addLayout(hbox)
        vbox.addWidget(self.export_postprocess_checkbox)
        vbox.addWidget(export_button, alignment=Qt.AlignLeft)
        layout.addLayout(vbox)

        gbox = QGroupBox()
        gbox.setTitle("Export")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)

    def create_large_stack_prediction_ui(self):
        stack_label = QLabel("Select your stack:")
        self.large_stack_textbox = QLineEdit()
        self.large_stack_textbox.setReadOnly(True)
        stack_button = QPushButton("Select...")
        stack_button.clicked.connect(self.select_stack)

        result_label = QLabel("Result Directory:")
        self.result_dir_textbox = QLineEdit()
        self.result_dir_textbox.setReadOnly(True)
        result_dir_button = QPushButton("Select...")
        result_dir_button.clicked.connect(self.select_result_dir)

        self.large_stack_info = QLabel("stack info")

        self.run_pipeline_button = QPushButton("Run Prediction")
        self.run_pipeline_button.setMinimumWidth(150)
        self.run_pipeline_button.clicked.connect(self.run_pipeline_over_large_stack)
        self.stop_pipeline_button = QPushButton("Stop")
        self.stop_pipeline_button.setMinimumWidth(150)
        self.stop_pipeline_button.setEnabled(False)
        self.stop_pipeline_button.clicked.connect(self.stop_pipeline)


        self.pipeline_progressbar = QProgressBar()

        # layout
        layout = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(stack_label)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.large_stack_textbox)
        hbox.addWidget(stack_button)
        vbox.addLayout(hbox)
        vbox.addWidget(result_label)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.result_dir_textbox)
        hbox.addWidget(result_dir_button)
        vbox.addLayout(hbox)
        vbox.addWidget(self.large_stack_info)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.run_pipeline_button, alignment=Qt.AlignLeft)
        hbox.addWidget(self.stop_pipeline_button, alignment=Qt.AlignLeft)
        vbox.addLayout(hbox)
        vbox.addWidget(self.pipeline_progressbar)
        layout.addLayout(vbox)

        gbox = QGroupBox()
        gbox.setTitle("Run Prediction Pipeline")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)

    def new_layer_checkbox_changed(self):
        state = self.new_layer_checkbox.checkState()
        self.prediction_layer_combo.setEnabled(state == Qt.Unchecked)
        self.seg_add_radiobutton.setEnabled(state == Qt.Unchecked)
        self.seg_replace_radiobutton.setEnabled(state == Qt.Unchecked)

    def check_input_layers(self, event: Event):
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
                layer.colormap = colormaps.create_colormap(10)[0]
                if "Segmentation" in layer.name:
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

    def clear_sam_auto_masks(self):
        self.sam_auto_masks = None

    def postprocess_layer_removed(self, event: Event):
        """Fires when current postprocess layer is removed."""
        if (
            self.postprocess_layer is not None and
            self.postprocess_layer not in self.viewer.layers
        ):
            self.postprocess_layer = None

    def sam_post_checked(self, checked: bool):
        if checked:
            self.sam_auto_post_checkbox.setChecked(False)

    def sam_auto_post_checked(self, checked: bool):
        if checked:
            self.sam_post_checkbox.setChecked(False)

    def select_storage(self):
        selected_file, _filter = QFileDialog.getOpenFileName(
            self, "Jug Lab", ".", "Feature Storage(*.hdf5)"
        )
        if selected_file is not None and len(selected_file) > 0:
            self.storage_textbox.setText(selected_file)
            # load the storage
            self.storage = h5py.File(selected_file, "r")
            # set the patch size and overlap from the selected storage
            self.patch_size = self.storage.attrs.get("patch_size", self.patch_size)
            self.overlap = self.storage.attrs.get(
                "overlap", self.overlap)
            self.stride, _ = get_stride_margin(self.patch_size, self.overlap)

            # initialize the model based on the selected storage
            img_height = self.storage.attrs["img_height"]
            img_width = self.storage.attrs["img_width"]
            # TODO: raise an error if current image dims are in conflicting with storage
            model_name = self.storage.attrs["model"]
            self.model_adapter = get_model(model_name, img_height, img_width)
            print(model_name, self.patch_size, self.overlap)

    def add_labels_layer(self):
        self.image_layer = get_layer(
            self.viewer,
            self.image_combo.currentText(), config.NAPARI_IMAGE_LAYER
        )
        if self.image_layer is None:
            notif.show_error("No Image layer is added or selected!")
            return

        layer = self.viewer.add_labels(
            np.zeros(get_stack_dims(self.image_layer.data), dtype=np.uint8),
            name="Labels", opacity=1.0
        )
        layer.colormap = colormaps.create_colormap(10)[0]
        layer.brush_size = 1

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
        # class zero is the napari background class.
        class_indices = [i for i in class_indices if i > 0]
        for class_idx in class_indices:
            positions = np.argwhere(layer.data == class_idx)
            labels_dict[class_idx] = positions

        return labels_dict

    def analyze_labels(self, labels_dict: dict = None):
        if labels_dict is None:
            labels_dict = self.get_class_labels()
        num_labels = [len(v) for v in labels_dict.values()]
        self.num_class_label.setText(f"Number of classes: {len(num_labels)}")
        each_class = "\n".join([
            f"class {i + 1}: {num_labels[i]:,d}" for i in range(len(num_labels))
        ])
        self.each_class_label.setText("Labels per class:\n" + each_class)

    def get_train_data(self):
        # get ground truth class labels
        labels_dict = self.get_class_labels()
        if len(labels_dict) == 0:
            return None
        if self.storage is None:
            notif.show_error("No embeddings storage file is selected!")
            return None
        # update labels stats
        self.analyze_labels(labels_dict)

        num_slices, img_height, img_width = get_stack_dims(self.image_layer.data)
        num_labels = sum([len(v) for v in labels_dict.values()])
        total_channels = self.model_adapter.get_total_output_channels()
        train_data = np.zeros((num_labels, total_channels))
        labels = np.zeros(num_labels, dtype="int32") - 1
        count = 0
        for class_index in np_progress(
            labels_dict, desc="getting training data", total=len(labels_dict.keys())
        ):
            class_label_coords = labels_dict[class_index]
            uniq_slices = np.unique(class_label_coords[:, 0]).tolist()
            # for each unique slice, load unique patches from the storage,
            # then get the pixel features within loaded patch.
            for slice_index in np_progress(uniq_slices, desc="reading slices"):
                slice_coords = class_label_coords[
                    class_label_coords[:, 0] == slice_index
                ][:, 1:]  # omit the slice dim
                patch_indices = get_patch_indices(
                    slice_coords, img_height, img_width,
                    self.patch_size, self.overlap
                )
                grp_key = str(slice_index)
                slice_dataset = self.storage[grp_key][self.model_adapter.name]
                for p_i in np.unique(patch_indices):
                    patch_coords = slice_coords[patch_indices == p_i]
                    patch_features = slice_dataset[p_i]
                    train_data[count: count + len(patch_coords)] = patch_features[
                        patch_coords[:, 0] % self.stride,
                        patch_coords[:, 1] % self.stride
                    ]
                    labels[
                        count: count + len(patch_coords)
                    ] = class_index - 1  # to have bg class as zero
                    count += len(patch_coords)

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
            class_weight="balanced",
            min_samples_split=15,
            min_samples_leaf=3,
            max_features=25,
            n_jobs=2 if os.cpu_count() < 5 else os.cpu_count() - 3,
            verbose=1
        )
        rf_classifier.fit(train_data, labels)
        self.rf_model = rf_classifier
        self.model_status_label.setText("Model status: Ready!")
        notif.show_info("Model status: Training is Done!")
        self.model_save_button.setEnabled(True)

    def load_rf_model(self):
        selected_file, _filter = QFileDialog.getOpenFileName(
            self, "Jug Lab", ".", "model(*.bin)"
        )
        if len(selected_file) > 0:
            # to suppress the sklearn InconsistentVersionWarning
            # which stops the napari (probably a napari bug).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # load the rf model
                with open(selected_file, mode="rb") as f:
                    model_data = pickle.load(f)
                # compatibility check for old format rf model
                if isinstance(model_data, dict):
                    # new format
                    self.rf_model = model_data["rf_model"]
                    # if there is no model_adapter loaded already
                    # (users can just load the rf model)
                    if self.model_adapter is None:
                        model_name = model_data["model_name"]
                        self.patch_size = model_data["patch_size"]
                        self.overlap = model_data["overlap"]
                        img_height = model_data["img_height"]
                        img_width = model_data["img_width"]
                        # init the model adapter
                        self.model_adapter = get_model(model_name, img_height, img_width)
                else:
                    # old format
                    self.rf_model = model_data

                notif.show_info("Model was loaded successfully.")
                self.model_status_label.setText("Model status: Ready!")
                self.model_save_button.setEnabled(True)

    def save_rf_model(self):
        if self.rf_model is None:
            notif.show_info("There is no trained model!")
            return
        selected_file, _filter = QFileDialog.getSaveFileName(
            self, "Jug Lab", ".", "model(*.bin)"
        )
        if len(selected_file) > 0:
            if not selected_file.endswith(".bin"):
                selected_file += ".bin"
            # save rf model along with metadata
            model_data = {
                "rf_model": self.rf_model,
                "model_name": self.model_adapter.name,
                "img_height": self.storage.attrs["img_height"],
                "img_width": self.storage.attrs["img_width"],
                "patch_size": self.patch_size,
                "overlap": self.overlap
            }
            with open(selected_file, mode="wb") as f:
                pickle.dump(model_data, f)
            notif.show_info("Model was saved successfully.")

    def predict(self, whole_stack=False):
        self.prediction_progress.setValue(0)
        if self.rf_model is None:
            notif.show_error("There is no trained RF model!")
            return

        self.image_layer = get_layer(
            self.viewer,
            self.image_combo.currentText(), config.NAPARI_IMAGE_LAYER
        )
        if self.image_layer is None:
            notif.show_error("No Image layer is selected!")
            return

        if self.storage is None:
            notif.show_error("No storage is selected!")
            return

        num_slices, img_height, img_width = get_stack_dims(self.image_layer.data)
        if self.new_layer_checkbox.checkState() == Qt.Checked:
            self.segmentation_layer = self.viewer.add_labels(
                np.zeros((num_slices, img_height, img_width), dtype=np.uint8),
                name="Segmentations"
            )
        else:
            # using selected layer for the segmentation
            self.segmentation_layer = get_layer(
                self.viewer,
                self.prediction_layer_combo.currentText(), config.NAPARI_LABELS_LAYER
            )
            if self.segmentation_layer is None:
                notif.show_error("No segmentation layer is selected!")
                return

        slice_indices = []
        if not whole_stack:
            # only predict the current slice
            slice_indices.append(self.viewer.dims.current_step[0])
        else:
            slice_indices = range(num_slices)

        if whole_stack:
            self.predict_stop_button.setEnabled(True)
        # run prediction in another thread
        self.predict_worker = create_worker(
            self.run_prediction, slice_indices, img_height, img_width
        )
        self.predict_worker.yielded.connect(self.update_prediction_progress)
        self.predict_worker.finished.connect(self.prediction_is_done)
        self.predict_worker.run()

    def run_prediction(self, slice_indices, img_height, img_width):
        for slice_index in np_progress(slice_indices):
            segmentation_image = self.predict_slice(
                self.rf_model, slice_index, img_height, img_width
            )
            # add/update segmentation result layer
            if (
                self.new_layer_checkbox.checkState() == Qt.Checked or
                self.seg_replace_radiobutton.isChecked()
            ):
                self.segmentation_layer.data[slice_index] = segmentation_image
            else:
                # add new result to the previous one (logical OR)
                old_bg = self.segmentation_layer.data[slice_index] == 0
                new_no_bg = segmentation_image > 0
                mask = np.logical_or(old_bg, new_no_bg)
                self.segmentation_layer.data[slice_index][mask] = segmentation_image[mask]

            yield (slice_index, len(slice_indices))

        if self.new_layer_checkbox.checkState() == Qt.Checked:
            cm, _ = colormaps.create_colormap(len(np.unique(segmentation_image)))
            self.segmentation_layer.colormap = cm
        self.segmentation_layer.refresh()

    def predict_slice(self, rf_model, slice_index, img_height, img_width):
        """Predict a slice patch by patch"""
        segmentation_image = []
        # shape: N x target_size x target_size x C
        feature_patches = self.storage[str(slice_index)][self.model_adapter.name][:]
        num_patches = feature_patches.shape[0]
        total_channels = self.model_adapter.get_total_output_channels()
        for i in np_progress(range(num_patches), desc="Predicting slice patches"):
            input_data = feature_patches[i].reshape(-1, total_channels)
            predictions = rf_model.predict(input_data).astype(np.uint8)
            # to match segmentation colormap with the labels' colormap
            predictions[predictions > 0] += 1
            segmentation_image.append(predictions)

        segmentation_image = np.vstack(segmentation_image)
        # reshape into the image size + padding
        patch_rows, patch_cols = get_num_patches(
            img_height, img_width, self.patch_size, self.overlap
        )
        segmentation_image = segmentation_image.reshape(
            patch_rows, patch_cols, self.stride, self.stride
        )
        segmentation_image = np.moveaxis(segmentation_image, 1, 2).reshape(
            patch_rows * self.stride,
            patch_cols * self.stride
        )
        # skip paddings
        segmentation_image = segmentation_image[:img_height, :img_width]

        return segmentation_image

    def stop_predict(self):
        if self.predict_worker is not None:
            self.predict_worker.quit()
            self.predict_worker = None
        self.predict_stop_button.setEnabled(False)

    def update_prediction_progress(self, values):
        curr, total = values
        self.prediction_progress.setMinimum(0)
        self.prediction_progress.setMaximum(total)
        self.prediction_progress.setValue(curr + 1)
        self.prediction_progress.setFormat("slice %v of %m (%p%)")

    def prediction_is_done(self):
        self.predict_all_button.setEnabled(True)
        self.predict_stop_button.setEnabled(False)
        print("Prediction is done!")
        notif.show_info("Prediction is done!")

    def get_postprocess_params(self):
        smoothing_iterations = 25
        if len(self.smoothing_iteration_textbox.text()) > 0:
            smoothing_iterations = int(self.smoothing_iteration_textbox.text())

        area_threshold = 0
        if len(self.area_threshold_textbox.text()) > 0:
            area_threshold = int(self.area_threshold_textbox.text())
        area_is_absolute = False
        if self.area_abs_radiobutton.isChecked():
            area_is_absolute = True

        return smoothing_iterations, area_threshold, area_is_absolute

    def postprocess_segmentation(self, whole_stack=False):
        self.segmentation_layer = get_layer(
            self.viewer,
            self.prediction_layer_combo.currentText(), config.NAPARI_LABELS_LAYER
        )
        if self.segmentation_layer is None:
            notif.show_error("No segmentation layer is selected!")
            return

        smoothing_iterations, area_threshold, area_is_absolute = self.get_postprocess_params()

        num_slices, img_height, img_width = get_stack_dims(self.image_layer.data)
        slice_indices = []
        if not whole_stack:
            # only predict the current slice
            slice_indices.append(self.viewer.dims.current_step[0])
        else:
            slice_indices = range(num_slices)

        if self.postprocess_layer is None:
            self.postprocess_layer = self.viewer.add_labels(
                data=np.zeros_like(self.segmentation_layer.data),
                name="Postprocessing"
            )
        self.postprocess_layer.colormap = self.segmentation_layer.colormap

        for slice_index in np_progress(slice_indices):
            prediction = self.segmentation_layer.data[slice_index]

            if self.sam_post_checkbox.isChecked():
                self.postprocess_layer.data[slice_index] = postprocess_with_sam(
                    prediction,
                    smoothing_iterations, area_threshold, area_is_absolute
                )
            elif self.sam_auto_post_checkbox.isChecked():
                # get input image/slice
                num_slices, _, _ = get_stack_dims(self.image_layer.data)
                input_image = self.image_layer.data
                if num_slices > 1:
                    input_image = self.image_layer.data[slice_index]
                iou_threshold = float(self.sam_auto_threshold_textbox.text())
                # get sam auto-segmentation masks
                if self.sam_auto_masks is None or whole_stack:
                    self.sam_auto_masks = get_sam_auto_masks(input_image)
                # postprocess
                self.postprocess_layer.data[slice_index] = postprocess_with_sam_auto(
                    self.sam_auto_masks,
                    prediction,
                    smoothing_iterations, iou_threshold,
                    area_threshold, area_is_absolute
                )
            else:
                self.postprocess_layer.data[slice_index] = postprocess(
                    prediction,
                    smoothing_iterations, area_threshold, area_is_absolute
                )

        self.postprocess_layer.refresh()

    def export_segmentation(self):
        if self.segmentation_layer is None:
            notif.show_error("No segmentation layer is selected!")
            return

        exporter = EXPORTERS[self.export_format_combo.currentText()]
        # export_format = self.export_format_combo.currentText()
        selected_file, _filter = QFileDialog.getSaveFileName(
            self, "Jug Lab", ".", f"Segmentation(*.{exporter.extension})"
        )
        if selected_file is None or len(selected_file) == 0:
            return  # user canceled export

        if not selected_file.endswith(f".{exporter.extension}"):
            selected_file += f".{exporter.extension}"
        layer_to_export = self.segmentation_layer
        if (
            self.export_postprocess_checkbox.isChecked() and
            self.postprocess_layer is not None
        ):
            layer_to_export = self.postprocess_layer

        exporter.export(layer_to_export, selected_file)

        notif.show_info("Selected layer was exported successfully.")

    def select_stack(self):
        selected_file, _filter = QFileDialog.getOpenFileName(
            self, "Jug Lab", ".", "TIFF stack (*.tiff *.tif)"
        )
        if selected_file is not None and len(selected_file) > 0:
            # get stack info
            with TiffFile(selected_file) as tiff_stack:
                axes = tiff_stack.series[0].axes
                assert ("Y" in axes) and ("X" in axes), "Could not find YX in the stack axes!"
                stack_dims = tiff_stack.series[0].shape
            stack_height = stack_dims[axes.index("Y")]
            stack_width = stack_dims[axes.index("X")]
            # selected stack image dimensions should be as the same as the current image
            self.image_layer = get_layer(
                self.viewer,
                self.image_combo.currentText(), config.NAPARI_IMAGE_LAYER
            )
            if self.image_layer is not None:
                _, img_height, img_width = get_stack_dims(self.image_layer.data)
            else:
                img_height, img_width = 0, 0
            if img_height != stack_height or img_width != stack_width:
                notif.show_error("Stack image dimensions do not match the current image!")
                return
            # tutto bene!
            self.large_stack_textbox.setText(selected_file)
            self.large_stack_info.setText(
                f"Axes: {axes}, Dims: {stack_dims}, DType: {tiff_stack.series[0].dtype}"
            )
            # set default result directory
            res_dir = Path(selected_file).parent
            self.result_dir_textbox.setText(str(res_dir.absolute()))

    def select_result_dir(self):
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Select a directory", self.large_stack_textbox.text(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if selected_dir is not None and len(selected_dir) > 0:
            self.result_dir_textbox.setText(selected_dir)

    def run_pipeline_over_large_stack(self):
        if self.large_stack_textbox.text() == "":
            notif.show_error("No TIFF Stack is selected!")
            return
        if self.rf_model is None:
            notif.show_error("No RF model is trained!")
            return

        self.pipeline_progressbar.setValue(0)
        self.run_pipeline_button.setEnabled(False)
        self.stop_pipeline_button.setEnabled(True)
        # create the slice temporary storage
        tmp_storage_path = Path.home().joinpath(".featureforest", "tmp_storage.h5")
        storage = h5py.File(tmp_storage_path, "w")
        storage_group = storage.create_group("slice")
        # run pipeline prediction in another thread
        self.pipeline_worker = create_worker(
            self.run_pipeline,
            Path(self.large_stack_textbox.text()),
            Path(self.result_dir_textbox.text()),
            storage_group
        )
        self.pipeline_worker.yielded.connect(self.update_pipeline_progress)
        self.pipeline_worker.finished.connect(self.pipeline_is_done)
        self.pipeline_worker.run()

    def run_pipeline(
            self, tiff_stack_file: str, result_dir: Path, storage_group: h5py.Group
    ):
        prediction_dir = result_dir.joinpath("predictions")
        prediction_dir.mkdir(parents=True, exist_ok=True)
        simple_post_dir = result_dir.joinpath("post_simple")
        simple_post_dir.mkdir(parents=True, exist_ok=True)
        sam_post_dir = result_dir.joinpath("post_sam")
        sam_post_dir.mkdir(parents=True, exist_ok=True)

        with TiffFile(tiff_stack_file) as tiff_stack:
            for page_idx, page in np_progress(
                enumerate(tiff_stack.pages),
                desc="runing the pipeline", total=len(tiff_stack.pages)
            ):
                image = page.asarray()
                prediction_mask = extract_predict(
                    image, self.model_adapter, storage_group, self.rf_model
                )
                # save the prediction
                tifffile.imwrite(
                    prediction_dir.joinpath(f"slice_{page_idx:04}_prediction.tiff"),
                    prediction_mask
                )
                # post-processing
                smoothing_iterations, area_threshold, area_is_absolute = self.get_postprocess_params()
                post_mask = postprocess(
                    prediction_mask, smoothing_iterations,
                    area_threshold, area_is_absolute
                )
                tifffile.imwrite(
                    simple_post_dir.joinpath(f"slice_{page_idx:04}_post_simple.tiff"),
                    post_mask
                )
                post_sam_mask = postprocess_with_sam(
                    prediction_mask,
                    smoothing_iterations, area_threshold, area_is_absolute
                )
                tifffile.imwrite(
                    sam_post_dir.joinpath(f"slice_{page_idx:04}_post_sam.tiff"),
                    post_sam_mask
                )

                yield page_idx, len(tiff_stack.pages)
        # closing the h5 storage & remove the file
        try:
            storage_group.file.close()
            self.remove_temp_storage()
        except Exception:  # noqa: BLE001
            pass

    def stop_pipeline(self):
        if self.pipeline_worker is not None:
            self.pipeline_worker.quit()
            self.pipeline_worker = None
        self.stop_pipeline_button.setEnabled(False)

    def update_pipeline_progress(self, values):
        curr, total = values
        self.pipeline_progressbar.setMinimum(0)
        self.pipeline_progressbar.setMaximum(total)
        self.pipeline_progressbar.setValue(curr + 1)
        self.pipeline_progressbar.setFormat("slice %v of %m (%p%)")

    def pipeline_is_done(self):
        self.run_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setEnabled(False)
        self.remove_temp_storage()
        print("Prediction is done!")
        notif.show_info("Prediction is done!")

    def remove_temp_storage(self):
        tmp_storage_path = Path.home().joinpath(".featureforest", "tmp_storage.h5")
        if tmp_storage_path.exists():
            tmp_storage_path.unlink()
