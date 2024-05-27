import pickle

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
import nrrd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import SAM
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
from .utils.postprocess import (
    postprocess_segmentation,
)
from .utils.postprocess_with_sam import postprocess_segmentations_with_sam


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
        self.patch_size = 512
        self.overlap = 384
        self.stride = self.patch_size - self.overlap

        self.prepare_widget()

        # init sam model & predictor
        self.sam_model, self.device = self.get_model_on_device()

    def closeEvent(self, event):
        print("closing")
        self.viewer.layers.events.inserted.disconnect(self.check_input_layers)
        self.viewer.layers.events.removed.disconnect(self.check_input_layers)
        self.viewer.layers.events.changed.disconnect(self.check_input_layers)

        self.viewer.layers.events.inserted.disconnect(self.check_label_layers)
        self.viewer.layers.events.removed.disconnect(self.check_label_layers)

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

        # self.sam_progress = QProgressBar()
        # self.save_storage_button = QPushButton("Save SAM Embeddings")
        # self.save_storage_button.clicked.connect(self.save_embeddings)
        # self.save_storage_button.setMinimumWidth(150)
        # self.save_storage_button.setMaximumWidth(150)
        # self.save_storage_button.setEnabled(False)

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
        self.seg_add_radiobutton.setChecked(True)
        self.seg_add_radiobutton.setEnabled(False)
        self.seg_replace_radiobutton = QRadioButton("Replace Segmentations")
        self.seg_replace_radiobutton.setEnabled(False)

        # post-process ui
        area_label = QLabel("Area Threshold(%):")
        self.area_threshold_textbox = QLineEdit()
        self.area_threshold_textbox.setText("15")
        self.area_threshold_textbox.setValidator(
            QDoubleValidator(
                1.000, 100.000, 3, notation=QDoubleValidator.StandardNotation
            )
        )
        self.area_threshold_textbox.setToolTip(
            "Keeps regions with area above the threshold percentage."
        )
        self.area_threshold_textbox.setEnabled(False)

        self.sam_post_checkbox = QCheckBox("Use SAM Predictor")
        self.sam_post_checkbox.setEnabled(False)

        self.postprocess_checkbox = QCheckBox("Postprocess Segmentations")
        self.postprocess_checkbox.stateChanged.connect(self.postprocess_checkbox_changed)

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

        export_nrrd_button = QPushButton("Export To nrrd")
        export_nrrd_button.setMinimumWidth(150)
        export_nrrd_button.clicked.connect(lambda: self.export_segmentation("nrrd"))

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
        vbox.addWidget(self.postprocess_checkbox)
        vbox.addWidget(area_label)
        vbox.addWidget(self.area_threshold_textbox)
        vbox.addWidget(self.sam_post_checkbox)
        vbox.addSpacing(20)
        vbox.addWidget(predict_button, alignment=Qt.AlignLeft)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.predict_all_button, alignment=Qt.AlignLeft)
        hbox.addWidget(self.predict_stop_button, alignment=Qt.AlignLeft)
        vbox.addLayout(hbox)
        vbox.addWidget(self.prediction_progress)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 15, 0, 0)
        hbox.addWidget(export_nrrd_button, alignment=Qt.AlignLeft)
        vbox.addLayout(hbox)
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
        self.seg_replace_radiobutton.setEnabled(state == Qt.Unchecked)

    def postprocess_checkbox_changed(self):
        state = self.postprocess_checkbox.checkState()
        self.area_threshold_textbox.setEnabled(state == Qt.Checked)
        self.sam_post_checkbox.setEnabled(state == Qt.Checked)

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

    def select_storage(self):
        selected_file, _filter = QFileDialog.getOpenFileName(
            self, "Jug Lab", ".", "Embeddings Storage(*.hdf5)"
        )
        if selected_file is not None and len(selected_file) > 0:
            self.storage_textbox.setText(selected_file)
            # load the storage
            self.storage = h5py.File(selected_file, "r")
            self.patch_size = self.storage.attrs.get("patch_size", self.patch_size)
            self.overlap = self.storage.attrs.get(
                "overlap", self.overlap)
            self.stride, _ = get_stride_margin(self.patch_size, self.overlap)

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

    def analyze_labels(self):
        labels_dict = self.get_class_labels()
        num_labels = [len(v) for v in labels_dict.values()]
        self.num_class_label.setText(f"Number of classes: {len(num_labels)}")
        each_class = "\n".join([
            f"class {i + 1}: {num_labels[i]:,d}" for i in range(len(num_labels))
        ])
        self.each_class_label.setText("Labels per class:\n" + each_class)

    def get_model_on_device(self):
        return SAM.setup_mobile_sam_model()

    def get_train_data(self):
        # get ground truth class labels
        labels_dict = self.get_class_labels()
        if not labels_dict:
            return None
        if self.storage is None:
            notif.show_error("No embeddings storage file is selected!")
            return None

        num_slices, img_height, img_width = get_stack_dims(self.image_layer.data)
        num_labels = sum([len(v) for v in labels_dict.values()])
        total_channels = SAM.ENCODER_OUT_CHANNELS + SAM.EMBED_PATCH_CHANNELS
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
                slice_dataset = self.storage[grp_key]["sam"]
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
            min_samples_leaf=1,
            n_jobs=2,
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
            with open(selected_file, mode="rb") as f:
                self.rf_model = pickle.load(f)
            notif.show_info("Model was loaded successfully.")
            self.model_status_label.setText("Model status: Ready!")

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
            with open(selected_file, mode="wb") as f:
                pickle.dump(self.rf_model, f)
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
        self.prediction_worker = create_worker(
            self.run_prediction, slice_indices, img_height, img_width
        )
        self.prediction_worker.yielded.connect(self.update_prediction_progress)
        self.prediction_worker.finished.connect(self.prediction_is_done)
        self.prediction_worker.run()

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
        feature_patches = self.storage[str(slice_index)]["sam"][:]
        num_patches = feature_patches.shape[0]
        total_channels = SAM.ENCODER_OUT_CHANNELS + SAM.EMBED_PATCH_CHANNELS
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

        # check for postprocessing
        if self.postprocess_checkbox.checkState() == Qt.Checked:
            # apply postprocessing
            area_threshold = None
            if len(self.area_threshold_textbox.text()) > 0:
                area_threshold = float(self.area_threshold_textbox.text()) / 100
            if self.sam_post_checkbox.checkState() == Qt.Checked:
                segmentation_image = postprocess_segmentations_with_sam(
                    self.sam_model, segmentation_image, area_threshold
                )
            else:
                segmentation_image = postprocess_segmentation(
                    segmentation_image, area_threshold
                )

        return segmentation_image

    def stop_predict(self):
        if self.prediction_worker is not None:
            self.prediction_worker.quit()
            self.prediction_worker = None
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

    def export_segmentation(self, out_format="nrrd"):
        if self.segmentation_layer is None:
            notif.show_error("No segmentation layer is selected!")
            return

        selected_file, _filter = QFileDialog.getSaveFileName(
            self, "Jug Lab", ".", "Segmentation(*.nrrd)"
        )
        if selected_file is not None and len(selected_file) > 0:
            if not selected_file.endswith(".nrrd"):
                selected_file += ".nrrd"
            nrrd.write(selected_file, np.transpose(self.segmentation_layer.data))
            notif.show_info("Selected segmentation was exported successfully.")
