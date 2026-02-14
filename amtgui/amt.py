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

from audio_control import AudioControl
from audio_player import AudioPlayer
from menu import MenuBar

class AMTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AMT")

        self.setMinimumSize(QSize(1100, 800))

        self.menu = MenuBar(self)

        # Mel spectrogram
        mel_spectrogram = QLabel("Mel Spectrogram")
        font = mel_spectrogram.font()
        font.setPointSize(30)
        mel_spectrogram.setFont(font)
        mel_spectrogram.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )

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

        # Settings
        settings = QLabel("Settings")
        font = settings.font()
        font.setPointSize(30)
        settings.setFont(font)
        settings.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )

        audioLayout = QVBoxLayout()
        audioLayout.addWidget(self.audio_control.audio_buttons)

        bottom = QHBoxLayout()
        bottom.addLayout(audioLayout)
        bottom.addWidget(self.audio_player.audio_control_buttons)
        bottom.addWidget(settings)

        layout = QVBoxLayout()
        layout.addWidget(mel_spectrogram)
        layout.addWidget(midi)

        layout.addLayout(bottom)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

app = QApplication(sys.argv)

window = AMTMainWindow()
window.show()

app.exec()
