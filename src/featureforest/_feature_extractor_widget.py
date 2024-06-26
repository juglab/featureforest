import napari
import napari.utils.notifications as notif
from napari.utils.events import Event
from napari.qt.threading import create_worker
from napari.utils import progress as np_progress

from qtpy.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QGroupBox,
    QPushButton, QLabel, QComboBox, QLineEdit,
    QFileDialog, QProgressBar,
)
from qtpy.QtCore import Qt

import h5py

from .widgets import (
    ScrollWidgetWrapper,
    get_layer,
)
from .models import get_available_models, get_model
from .utils import (
    config
)
from .utils.data import (
    get_stack_dims,
)
from .utils.extract import (
    get_slice_features
)


class FeatureExtractorWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.extract_worker = None
        self.storage = None
        self.model_adapter = None
        self.device = None

        self.prepare_widget()

    def prepare_widget(self):
        self.base_layout = QVBoxLayout()
        scroll_content = QWidget()
        scroll_content.setLayout(self.base_layout)
        scroll = ScrollWidgetWrapper(scroll_content)
        vbox = QVBoxLayout()
        vbox.addWidget(scroll)
        self.setLayout(vbox)

        # input layer
        input_label = QLabel("Image Layer:")
        self.image_combo = QComboBox()
        # model selection
        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        self.model_combo.addItems(get_available_models())
        self.model_combo.setCurrentIndex(0)
        # sam storage
        storage_label = QLabel("Features Storage File:")
        self.storage_textbox = QLineEdit()
        self.storage_textbox.setReadOnly(True)
        storage_button = QPushButton("Set Storage File")
        storage_button.clicked.connect(self.save_storage)
        # extract button
        self.extract_button = QPushButton("Extract Features")
        self.extract_button.setEnabled(False)
        self.extract_button.clicked.connect(self.extract_embeddings)
        self.extract_button.setMinimumWidth(150)
        # stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_extracting)
        self.stop_button.setMinimumWidth(150)
        # progress
        self.stack_progress = QProgressBar()

        self.viewer.layers.events.inserted.connect(self.check_input_layers)
        self.viewer.layers.events.removed.connect(self.check_input_layers)
        self.viewer.layers.events.changed.connect(self.check_input_layers)
        self.check_input_layers(None)

        # layout
        layout = QVBoxLayout()
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(input_label)
        vbox.addWidget(self.image_combo)
        vbox.addWidget(self.model_combo)
        layout.addLayout(vbox)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(storage_label)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.storage_textbox)
        hbox.addWidget(storage_button)
        vbox.addLayout(hbox)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.extract_button, alignment=Qt.AlignLeft)
        hbox.addWidget(self.stop_button, alignment=Qt.AlignLeft)
        vbox.addLayout(hbox)
        layout.addLayout(vbox)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 5, 0, 0)
        vbox.addWidget(self.stack_progress)
        layout.addLayout(vbox)

        gbox = QGroupBox()
        gbox.setTitle("Feature Extractor Widget")
        gbox.setMinimumWidth(100)
        gbox.setLayout(layout)
        self.base_layout.addWidget(gbox)
        self.base_layout.addStretch(1)

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

    def save_storage(self):
        # default storage name
        image_layer_name = self.image_combo.currentText()
        model_name = self.model_combo.currentText()
        storage_name = f"{image_layer_name}_{model_name}.hdf5"
        # open the save dialog
        selected_file, _filter = QFileDialog.getSaveFileName(
            self, "Jug Lab", storage_name, "Embeddings Storage(*.hdf5)"
        )
        if selected_file is not None and len(selected_file) > 0:
            if not selected_file.endswith(".hdf5"):
                selected_file += ".hdf5"
            self.storage_textbox.setText(selected_file)
            self.extract_button.setEnabled(True)

    def extract_embeddings(self):
        self.stack_progress.setValue(0)
        # check image layer
        image_layer_name = self.image_combo.currentText()
        image_layer = get_layer(
            self.viewer, image_layer_name, config.NAPARI_IMAGE_LAYER
        )
        if image_layer is None:
            notif.show_error("No Image layer is selected.")
            return
        # check storage
        storage_path = self.storage_textbox.text()
        if storage_path is None or len(storage_path) < 6:
            notif.show_error("No storage path was set.")
            return
        # initialize the selected model
        _, img_height, img_width = get_stack_dims(image_layer.data)
        model_name = self.model_combo.currentText()
        self.model_adapter, self.device = get_model(model_name, img_height, img_width)

        self.extract_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.extract_worker = create_worker(
            self.get_stack_sam_embeddings,
            image_layer, storage_path
        )
        self.extract_worker.yielded.connect(self.update_extract_progress)
        self.extract_worker.finished.connect(self.extract_is_done)
        self.extract_worker.run()

    def get_stack_sam_embeddings(
        self, image_layer, storage_path
    ):
        # prepare the storage hdf5 file
        self.storage = h5py.File(storage_path, "w")
        # get sam embeddings slice by slice and save them into storage file
        num_slices, img_height, img_width = get_stack_dims(image_layer.data)
        self.storage.attrs["num_slices"] = num_slices
        self.storage.attrs["img_height"] = img_height
        self.storage.attrs["img_width"] = img_width
        self.storage.attrs["model"] = self.model_combo.currentText()
        self.storage.attrs["patch_size"] = self.model_adapter.patch_size
        self.storage.attrs["overlap"] = self.model_adapter.overlap

        for slice_index in np_progress(
            range(num_slices), desc="extract features for slices"
        ):
            image = image_layer.data[slice_index] if num_slices > 1 else image_layer.data
            slice_grp = self.storage.create_group(str(slice_index))
            get_slice_features(
                image, self.model_adapter.patch_size, self.model_adapter.overlap,
                self.model_adapter, self.device, slice_grp
            )

            yield (slice_index, num_slices)

        self.storage.close()

    def stop_extracting(self):
        if self.extract_worker is not None:
            self.extract_worker.quit()
            self.extract_worker = None
        if isinstance(self.storage, h5py.File):
            self.storage.close()
            self.storage = None
        self.stop_button.setEnabled(False)

    def update_extract_progress(self, values):
        curr, total = values
        self.stack_progress.setMinimum(0)
        self.stack_progress.setMaximum(total)
        self.stack_progress.setValue(curr + 1)
        self.stack_progress.setFormat("slice %v of %m (%p%)")

    def extract_is_done(self):
        self.extract_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print("Extracting is done!")
        notif.show_info("Extracting is done!")
