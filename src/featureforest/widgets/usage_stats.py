from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QDialog,
    QFileDialog, QMessageBox
)

from featureforest.utils.usage_stats import SegmentationUsageStats


class UsageStats(QDialog):
    def __init__(self, stats: SegmentationUsageStats) -> None:
        super().__init__()
        self.setModal(True)
        self.stats = stats
        # widgets
        self.info_label = QLabel()
        save_button = QPushButton("Save as CSV")
        save_button.setMinimumWidth(150)
        save_button.clicked.connect(self.save_stats)
        save_button.setDefault(True)
        reset_button = QPushButton("Reset")
        reset_button.setMinimumWidth(150)
        reset_button.clicked.connect(self.reset)
        # layouts
        layout = QVBoxLayout()
        layout.setSpacing(1)
        layout.addWidget(self.info_label)
        hbox = QHBoxLayout()
        hbox.addWidget(save_button, alignment=Qt.AlignHCenter)
        hbox.addWidget(reset_button, alignment=Qt.AlignHCenter)
        layout.addLayout(hbox)
        self.setLayout(layout)
        self.setMinimumWidth(500)

        self.show_stats()

    def show_stats(self):
        df = self.stats.to_dataframe()
        self.info_label.setText(df.to_html(index=False))

    def save_stats(self):
        selected_file, _filter = QFileDialog.getSaveFileName(
            self, "FeatureForest", ".", "CSV file(*.csv)"
        )
        if selected_file is not None and len(selected_file) > 0:
            self.stats.save_as_csv(selected_file)
            self.close()

    def reset(self):
        confirmed = QMessageBox.question(self, "Confirm Reset", "Are you sure?")
        if confirmed == QMessageBox.Yes:
            self.stats.reset()
            self.show_stats()
            # self.info_label.setText("Reset!")




if __name__ == "__main__":
    import sys
    from qtpy.QtWidgets import QApplication


    stats = SegmentationUsageStats()

    app = QApplication(sys.argv)
    widget = UsageStats(stats)
    widget.show()

    sys.exit(app.exec_())
