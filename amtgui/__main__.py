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
from amtgui.pianoroll import PianoRollWidget
from amtgui.transcription import Transcription

class AMTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AMT")

        self.setMinimumSize(QSize(1100, 800))

        self.menu = MenuBar(self)

        self.mel_spectrogram_widget = MelSpectrogramWidget()
        self.pianoroll_widget = PianoRollWidget()

        # Audio buttons
        self.audio_control = AudioControl()
        self.audio_player = AudioPlayer()
        self.audio_control.audio_loaded.connect(self.audio_player.load_audio)
        self.audio_control.audio_loaded.connect(self.mel_spectrogram_widget.load_audio)
        self.audio_control.audio_loaded.connect(self.on_audio_loaded)
        self.audio_control.audio_reset.connect(self.mel_spectrogram_widget.reset_plot)
        self.audio_control.audio_reset.connect(self.pianoroll_widget.reset_plot)
        self.audio_control.audio_reset.connect(self.audio_player.reset_audio)

        # Settings
        self.model_selector = ModelSelector()
        self.model_selector.combo_box.currentTextChanged.connect(self.transcribe_audio)

        # Layout
        audioLayout = QVBoxLayout()
        audioLayout.addWidget(self.audio_control.audio_buttons)

        bottom = QHBoxLayout()
        bottom.addLayout(audioLayout)
        bottom.addWidget(self.audio_player.audio_control_buttons)
        bottom.addWidget(self.model_selector)

        layout = QVBoxLayout()
        layout.addWidget(self.mel_spectrogram_widget, 1)
        layout.addWidget(self.pianoroll_widget, 1)

        layout.addLayout(bottom)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.current_audio_path = None
        self.transcription_worker = None

    def on_audio_loaded(self, audio_path):
        self.current_audio_path = audio_path
        self.transcribe_audio()

    def transcribe_audio(self, _=None):
        if not self.current_audio_path:
            return

        model_name = self.model_selector.current_model()
        if model_name == "Select a model":
            return
        
        if not hasattr(self.model_selector, 'model') or self.model_selector.model is None:
            return

        print(f"Transcribing {self.current_audio_path} with model: {model_name}")
        self.pianoroll_widget.show_loading()
        
        self.transcription_worker = Transcription(self.model_selector.model, model_name, self.current_audio_path)
        self.transcription_worker.finished.connect(self.pianoroll_widget.plot_pianoroll)
        self.transcription_worker.start()

app = QApplication(sys.argv)

window = AMTMainWindow()
window.show()

app.exec()
