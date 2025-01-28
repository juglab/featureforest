import time

import pandas as pd



def format_seconds(total_seconds):
    minutes, seconds = divmod(total_seconds, 60)
    nice = f"{int(minutes)} minutes and {int(seconds)} seconds"
    return nice, int(minutes), int(seconds)


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
        # if the same old shhh :)
        if self.label_layer == layer:
            return
        # release the old layer event
        if self.label_layer is not None:
            self.label_layer.events.labels_update.disconnect()
        # set the new one
        self.label_layer = layer
        self.label_layer.events.labels_update.connect(self._on_label_updated)

    def set_file_path(self, path):
        self.file_path = path

    def training_started(self):
        self._training_start = time.perf_counter()

    def count_training(self):
        self.last_training_time = time.perf_counter() - self._training_start
        self.num_training += 1
        old_avg = self.avg_training_time
        self.avg_training_time = (
            old_avg * (self.num_training - 1) + self.last_training_time) / self.num_training
        # save the stats
        self.save_as_csv()

    def prediction_started(self):
        self._prediction_start = time.perf_counter()

    def count_prediction(self):
        self.num_prediction += 1
        self.prediction_time = time.perf_counter() - self._prediction_start
        # save the stats
        self.save_as_csv()

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
        # save the stats
        self.save_as_csv()

    def to_dataframe(self):
        data = {}
        data["labeling"] = format_seconds(self.last_label_on - self.labeling_start)[0]
        data["last_training"] = round(self.last_training_time, 2)
        data["average_training"] = round(self.avg_training_time, 2)
        data["num_trainings"] = self.num_training
        data["prediction_time"] = round(self.prediction_time, 2)
        data["num_predictions"] = self.num_prediction

        return pd.DataFrame(data, index=[0])

    def save_as_csv(self, file_path=None):
        where_to_save = self.file_path
        if file_path is not None:
            where_to_save = file_path
        if where_to_save is not None:
            df = self.to_dataframe()
            df.to_csv(where_to_save, index=False)
            print(f"Plugin usage stats is saved at {where_to_save}")
        else:
            print("Don't know where to save it!")
