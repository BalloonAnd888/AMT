from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from amtgui.model_loader import ModelManager

class ModelSelector(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)

        self.label = QLabel("Transcription Model")
        self.label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        self.model_manager = ModelManager()

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
        self.model = self.model_manager.load_model(model_name)

    def current_model(self):
        return self.combo_box.currentText()