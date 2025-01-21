import time


class SegmentationUsageStats:
    def __init__(self):
        self.labeling_start = 0
        self.last_label_on = 0
        self.num_training = 0
        self.last_training_time = 0
        self.avg_training_time = 0
        self.num_prediction = 0
        self.prediction_time = 0
        self.label_layer = None  # napari active labeling layer
        self.file_path = ""
        self._training_start = 0
        self._prediction_start = 0

    def set_label_layer(self, layer):
        # same old shhh :)
        if self.label_layer == layer:
            return
        # release the old layer event
        if self.label_layer is not None:
            self.label_layer.events.label_update.disconnect()
        # set the new one
        self.label_layer = layer
        self.label_layer.events.labels_update.connect(self._on_label_updated)

    def training_started(self):
        self._training_start = time.perf_counter()

    def count_training(self):
        self.last_training_time = time.perf_counter() - self._training_start
        self.num_training += 1
        old_avg = self.avg_training_time
        self.avg_training_time = (
            old_avg * (self.num_training - 1) + self.last_training_time) / self.num_training

    def prediction_started(self):
        self._prediction_start = time.perf_counter()

    def count_prediction(self):
        self.num_prediction += 1
        self.prediction_time = time.perf_counter() - self._prediction_start

    def reset(self):
        self.labeling_start = 0
        self.last_label_on = 0
        self.num_training = 0
        self.last_training_time = 0
        self.avg_training_time = 0
        self.num_prediction = 0
        self.prediction_time = 0
        # self.label_layer = None
        # self.file_path = ""
        self._training_start = 0
        self._prediction_start = 0

    def _on_label_updated(self):
        if self.labeling_start == 0:
            self.labeling_start = time.perf_counter()
        self.last_label_on = time.perf_counter()

    def format_seconds(self, seconds):
        pass

    def to_dataframe(self):
        pass
