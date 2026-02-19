from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

class ModelSelector(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)

        self.label = QLabel("Transcription Model")
        self.label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.combo_box = QComboBox()
        self.combo_box.addItems(
            ["Select a model", "End to End", "Onsets and Velocities", "Onsets and Frames"]
        )
        self.combo_box.currentTextChanged.connect(self.load_model)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.combo_box)

    def load_model(self, model_name):
        if model_name == "Select a model":
            return
        print(f"Loading model: {model_name}")

    def current_model(self):
        return self.combo_box.currentText()