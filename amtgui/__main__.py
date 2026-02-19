import sys 

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    )

from amtgui.audio_control import AudioControl
from amtgui.audio_player import AudioPlayer
from amtgui.menu import MenuBar
from amtgui.spectrogram import MelSpectrogramWidget
from amtgui.model_selector import ModelSelector

class AMTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AMT")

        self.setMinimumSize(QSize(1100, 800))

        self.menu = MenuBar(self)

        self.mel_spectrogram_widget = MelSpectrogramWidget()

        # MIDI 
        midi = QLabel("MIDI")
        font = midi.font()
        font.setPointSize(30)
        midi.setFont(font)
        midi.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )

        # Audio buttons
        self.audio_control = AudioControl()
        self.audio_player = AudioPlayer()
        self.audio_control.audio_loaded.connect(self.audio_player.load_audio)
        self.audio_control.audio_loaded.connect(self.mel_spectrogram_widget.load_audio)
        self.audio_control.audio_reset.connect(self.mel_spectrogram_widget.reset_plot)
        self.audio_control.audio_reset.connect(self.audio_player.reset_audio)
        self.audio_control.audio_loaded.connect(self.transcribe_audio)

        # Settings
        self.model_selector = ModelSelector()

        audioLayout = QVBoxLayout()
        audioLayout.addWidget(self.audio_control.audio_buttons)

        bottom = QHBoxLayout()
        bottom.addLayout(audioLayout)
        bottom.addWidget(self.audio_player.audio_control_buttons)
        bottom.addWidget(self.model_selector)

        layout = QVBoxLayout()
        layout.addWidget(self.mel_spectrogram_widget, 1)
        layout.addWidget(midi)

        layout.addLayout(bottom)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def transcribe_audio(self, audio_path):
        model_name = self.model_selector.current_model()
        if model_name == "Select a model":
            return
        print(f"Transcribing {audio_path} with model: {model_name}")

app = QApplication(sys.argv)

window = AMTMainWindow()
window.show()

app.exec()
