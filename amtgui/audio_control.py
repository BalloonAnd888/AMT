from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QPushButton, 
    QWidget,
    QVBoxLayout,
    )

class AudioControl(QObject):
    audio_loaded = Signal(str)

    def __init__(self):
        super().__init__()

        # Audio buttons
        self.load_audio_button = QPushButton("Load Audio")
        self.load_audio_button.pressed.connect(self.load_audio_button_clicked)

        self.record_audio_button = QPushButton("Record Audio")
        self.record_audio_button.pressed.connect(self.record_audio_button_clicked)

        self.reset_audio_button = QPushButton("Reset Audio")
        self.reset_audio_button.pressed.connect(self.reset_audio_button_clicked)

        self.audio_buttons_layout = QVBoxLayout()
        self.audio_buttons_layout.addWidget(self.load_audio_button)
        self.audio_buttons_layout.addWidget(self.record_audio_button)
        self.audio_buttons_layout.addWidget(self.reset_audio_button)

        self.audio_buttons = QWidget()
        self.audio_buttons.setLayout(self.audio_buttons_layout)

    def load_audio_button_clicked(self):
        print("Load Audio button clicked")

        audio_file_name, _ = QFileDialog.getOpenFileName(
            self.audio_buttons,
            "Load Audio",
            "",
            "Audio Files (*.wav *.mp3)"
        )

        if audio_file_name:
            print(f"File selected: {audio_file_name}")
            self.audio_loaded.emit(audio_file_name)
        
    def record_audio_button_clicked(self):
        print("Record Audio button clicked")
    
    def reset_audio_button_clicked(self):
        print("Reset Audio button clicked")
        
        button = QMessageBox.question(
            self.audio_buttons,
            "Reset Audio",
            "Are you sure you want to reset the audio?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if button == QMessageBox.StandardButton.Yes:
            print("Audio reset")
        else:
            print("Audio not reset")
