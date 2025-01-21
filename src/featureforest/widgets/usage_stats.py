from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QCheckBox, QRadioButton,
    QPushButton, QLabel, QDialog, QLineEdit,
    QFileDialog, QProgressBar,
)

from featureforest.utils.usage_stats import SegmentationUsageStats


class UsageStats(QDialog):
    def __init__(self, stats: SegmentationUsageStats) -> None:
        super().__init__()
        self.stats = stats
        # widgets & layout
        info_label = QLabel()
        save_button = QPushButton("Save as CSV")
        save_button.setMinimumWidth(150)
        save_button.clicked.connect(self.save_stats)
        reset_button = QPushButton("Reset")
        reset_button.setMinimumWidth(150)
        reset_button.clicked.connect(self.reset)

        layout = QVBoxLayout()
        layout.setSpacing(1)
        layout.addWidget(info_label)
        hbox = QHBoxLayout()
        hbox.addWidget(save_button, alignment=Qt.AlignHCenter)
        hbox.addWidget(reset_button, alignment=Qt.AlignHCenter)
        layout.addLayout(hbox)
        self.setLayout(layout)
        self.setMinimumWidth(500)

    def save_stats(self):
        pass

    def reset(self):
        pass




if __name__ == "__main__":
    import sys
    from qtpy.QtWidgets import QApplication


    stats = SegmentationUsageStats()

    app = QApplication(sys.argv)
    widget = UsageStats(stats)
    widget.show()

    sys.exit(app.exec_())
