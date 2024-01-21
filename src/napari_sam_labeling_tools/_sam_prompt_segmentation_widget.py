import napari
import napari.utils.notifications as notif
from napari.utils.events import Event
from napari.qt.threading import create_worker
from napari.utils import progress as np_progress
from napari.utils import Colormap

from qtpy.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QGroupBox, QCheckBox, QRadioButton,
    QPushButton, QLabel, QComboBox, QLineEdit,
    QFileDialog, QProgressBar,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator

import h5py
import nrrd
import numpy as np

from . import SAM
from .widgets import (
    ScrollWidgetWrapper,
    get_layer,
)
from .utils.data import TARGET_PATCH_SIZE, get_patch_position
from .utils import (
    colormaps, config
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
        self.segmentation_layer = None
        self.prompts_mask = None
        self.is_prompt_changed = True           # similarity matrix
        self.storage = None
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

        self.viewer.layers.events.inserted.connect(self.check_label_layers)
        self.viewer.layers.events.removed.connect(self.check_label_layers)
        self.check_label_layers(None)

    def get_model_on_device(self):
        return SAM.setup_lighthq_sam_model()

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
        add_point_prompt_button.clicked.connect(lambda: self.add_prompts_layer("point"))
        add_point_prompt_button.setMinimumWidth(150)
        add_box_prompt_button = QPushButton("Add Box Layer")
        add_box_prompt_button.clicked.connect(lambda: self.add_prompts_layer("box"))
        add_box_prompt_button.setMinimumWidth(150)
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
        hbox = QHBoxLayout()
        hbox.addWidget(add_point_prompt_button, alignment=Qt.AlignLeft)
        hbox.addWidget(add_box_prompt_button, alignment=Qt.AlignLeft)
        vbox.addLayout(hbox)
        layout.addLayout(vbox)

        gbox = QGroupBox()
        gbox.setTitle("Inputs")
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

        similarity_label = QLabel("Similarity Threshold:")
        self.similarity_threshold_textbox = QLineEdit()
        self.similarity_threshold_textbox.setText("0.91")
        self.similarity_threshold_textbox.setValidator(
            QDoubleValidator(0.000, 1.000, 3, notation=QDoubleValidator.StandardNotation)
        )
        self.similarity_threshold_textbox.setToolTip(
            "Keeps regions having cosine similarity with the prompt above the threshold"
            " (min=0.0, max=1.0)"
        )

        self.show_intermediate_checkbox = QCheckBox("Show Intermediate Results")
        # self.show_intermediate_checkbox.setChecked(True)

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
        vbox.addWidget(similarity_label)
        vbox.addWidget(self.similarity_threshold_textbox)
        vbox.addWidget(self.show_intermediate_checkbox)
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

    def check_label_layers(self, event: Event):
        pred_curr_text = self.prediction_layer_combo.currentText()
        self.prediction_layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, config.NAPARI_LABELS_LAYER):
                # to handle layer's name change by user
                layer.events.name.disconnect()
                layer.events.name.connect(self.check_label_layers)
                if "Segmentation" in layer.name:
                    self.prediction_layer_combo.addItem(layer.name)
        # put back the selected layers, if not removed
        if len(pred_curr_text) > 0:
            index = self.prediction_layer_combo.findText(
                pred_curr_text, Qt.MatchExactly
            )
            if index > -1:
                self.prediction_layer_combo.setCurrentIndex(index)

    def add_prompts_layer(self, prompt_type: str = "point"):
        layer = None
        if prompt_type == "point":
            layer = self.viewer.add_points(
                name="Point Prompts", ndim=3,
                face_color="lime", edge_width=0, size=11, opacity=0.85
            )
        else:  # box prompt
            layer = self.viewer.add_shapes(
                name="Box Prompts", ndim=3,
                face_color="#ffffff00", edge_color="lime", edge_width=3, opacity=0.85
            )
        layer.events.data.connect(self.prompt_changed)

    def prompt_changed(self, event):
        if isinstance(event, Event) and \
                event.source.name == self.prompt_combo.currentText():
            # user added, removed or moved points
            self.is_prompt_changed = True
        else:
            # combobox changed
            self.is_prompt_changed = True

    def select_storage(self):
        selected_file, _filter = QFileDialog.getOpenFileName(
            self, "Jug Lab", ".", "Embeddings Storage(*.hdf5)"
        )
        if selected_file is not None and len(selected_file) > 0:
            self.storage_textbox.setText(selected_file)
            # load the storage
            self.storage = h5py.File(selected_file, "r")

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

    def get_prompt_labels(self, num_slices, img_height, img_width):
        user_prompts = self.get_user_prompts()
        if user_prompts is None:
            return None
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
                input_image = np.repeat(
                    self.image_layer.data[slice_index, :, :, np.newaxis],
                    3, axis=-1
                )
                self.sam_predictor.set_image(input_image)
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
                input_image = np.repeat(
                    self.image_layer.data[slice_index, :, :, np.newaxis],
                    3, axis=-1
                )
                self.sam_predictor.set_image(input_image)
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

    def get_similarity_matrix(self, prompts_mask, curr_slice):
        """
        Calculate Cosine similarity for all pixels with prompts' mask average vector
        (in sam's embedding space).
        """
        prompt_avg_vector = np.zeros(SAM.EMBEDDING_SIZE + SAM.PATCH_CHANNELS)
        prompt_mask_positions = np.argwhere(prompts_mask == 1)
        for index in np.unique(prompt_mask_positions[:, 0]):
            slice_dataset = self.storage[str(index)]["sam"]
            coords_in_slice = prompt_mask_positions[:, 0] == index
            for pos in prompt_mask_positions[coords_in_slice]:
                y = pos[1]
                x = pos[2]
                # get patch including pixel position
                patch_row, patch_col = get_patch_position(y, x)
                patch_features = slice_dataset[patch_row, patch_col]
                # sum pixel embeddings
                prompt_avg_vector += patch_features[
                    y % TARGET_PATCH_SIZE, x % TARGET_PATCH_SIZE
                ]
        prompt_avg_vector /= len(prompt_mask_positions)

        # shape: patch_rows x patch_cols x target_size x target_size x C
        curr_slice_features = self.storage[str(curr_slice)]["sam"][:]
        patch_rows, patch_cols = curr_slice_features.shape[:2]
        # reshape it to the image size + padding
        curr_slice_features = curr_slice_features.transpose([0, 2, 1, 3, 4]).reshape(
            patch_rows * TARGET_PATCH_SIZE,
            patch_cols * TARGET_PATCH_SIZE,
            -1
        )
        # skip paddings
        _, img_height, img_width = self.image_layer.data.shape
        curr_slice_features = curr_slice_features[:img_height, :img_width]
        # calc. cosine similarity
        sim_mat = np.dot(curr_slice_features, prompt_avg_vector)
        sim_mat /= (
            np.linalg.norm(curr_slice_features, axis=2) *
            np.linalg.norm(prompt_avg_vector)
        )
        # scale similarity to the range of [0, 1]
        sim_mat = (sim_mat - sim_mat.min()) / (sim_mat.max() - sim_mat.min())
        # smooth out the similarity matrix
        sim_mat = process_similarity_matrix(sim_mat)

        return sim_mat

    def predict(self, whole_stack=False):
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

        if self.new_layer_checkbox.checkState() == Qt.Checked:
            self.segmentation_layer = self.viewer.add_labels(
                np.zeros(self.image_layer.data.shape, dtype=np.uint8),
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

        num_slices, img_height, img_width = self.image_layer.data.shape

        # get user prompt mask and calculate similarity matrix
        if self.is_prompt_changed:
            self.prompts_mask = self.get_prompt_labels(
                num_slices, img_height, img_width
            )
            if self.prompts_mask is None:
                return
            if self.prompts_mask.sum() == 0:
                print("SAM model couldn't generate any mask for the given prompts!")
                notif.show_warning(
                    "SAM model couldn't generate any mask for the given prompts!"
                )
                return

            self.is_prompt_changed = False
            if self.show_intermediate_checkbox.checkState() == Qt.Checked:
                # add sam predictor result's layer
                layer = self.viewer.add_labels(
                    data=self.prompts_mask, name="Prompt Labels", opacity=0.55
                )
                layer.colormap = Colormap(
                    np.array([[0.73, 0.48, 0.75, 1.0]])
                )

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
            self.run_prediction, slice_indices, self.prompts_mask
        )
        self.prediction_worker.yielded.connect(self.update_prediction_progress)
        self.prediction_worker.finished.connect(self.prediction_is_done)
        self.prediction_worker.run()

    def run_prediction(self, slice_indices, prompts_mask):
        for slice_index in np_progress(slice_indices):
            sim_mat = self.get_similarity_matrix(prompts_mask, slice_index)
            high_sim_mask = np.zeros_like(sim_mat, dtype=np.uint8)
            high_sim_mask[
                sim_mat >= float(self.similarity_threshold_textbox.text())
            ] = 255
            post_high_sim_mask = postprocess_label(high_sim_mask, 0.15)
            positive_prompts = generate_mask_prompts(post_high_sim_mask)

            if len(positive_prompts) == 0:
                print("No prompt was generated!")
                notif.show_warning(
                    f"No prompt was generated for slice {slice_index}!"
                    "Try a lower 'Similarity Threshold'."
                )
                self.is_prompt_changed = True
                continue

            # add user point prompts to the generated ones (y,x -> x,y).
            user_prompts = self.get_user_prompts()
            if len(user_prompts[0]) < 4:
                positive_prompts.extend([
                    tuple(p[1:][[1, 0]]) for p in self.get_user_prompts()
                    if p[0] == slice_index
                ])
            positive_prompts = np.array(positive_prompts)

            # add intermediate results only for one slice prediction
            if (
                len(slice_indices) == 1 and
                self.show_intermediate_checkbox.checkState() == Qt.Checked
            ):
                self.viewer.add_image(
                    data=sim_mat, name="Similiraty Matrix",
                    colormap="viridis", opacity=0.7, visible=False
                )
                self.viewer.add_image(
                    data=post_high_sim_mask, name="High Similiraty Mask",
                    colormap="gray", opacity=0.7, blending="additive", visible=False
                )
                self.viewer.add_points(
                    data=positive_prompts[:, [1, 0]], name="Generated Prompts",
                    face_color="lightseagreen", edge_color="white", edge_width=0,
                    opacity=0.7, size=8, visible=False
                )

            segmentation_image = self.predict_slice(slice_index, positive_prompts)
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
        # move the segmentation layer to the top
        old_idx = self.viewer.layers.index(self.segmentation_layer)
        if old_idx:
            self.viewer.layers.move(old_idx, len(self.viewer.layers))

    def predict_slice(self, slice_index, point_prompts):
        sam_masks = []
        input_image = np.repeat(
            self.image_layer.data[slice_index, :, :, np.newaxis],
            3, axis=-1
        )
        self.sam_predictor.set_image(input_image)
        for _, point in enumerate(point_prompts):
            point_labels = np.array([1])
            point_coords = point[np.newaxis, :]
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=True,
                hq_token_only=False,
            )
            sam_masks.append(masks[0])
            # sam_masks.extend(masks)

        sam_masks = np.array(sam_masks)
        segmentation_img = np.bitwise_or.reduce(sam_masks, axis=0)

        return segmentation_img

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
            nrrd.write(selected_file, self.segmentation_layer.data)
            notif.show_info("Selected segmentation was saved successfully.")