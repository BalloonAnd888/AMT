import librosa
import librosa.display
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from preprocessing.mel import MelSpectrogram
from preprocessing.constants import SAMPLE_RATE, HOP_LENGTH, MEL_FMIN, MEL_FMAX

class MelSpectrogramWorker(QThread):
    finished = Signal(object, object, object)

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            # Load audio with the project's sample rate
            y, sr = librosa.load(self.audio_path, sr=SAMPLE_RATE)
            
            # Convert to tensor
            audio = torch.from_numpy(y).float()
            
            # Compute Mel Spectrogram
            mel_extractor = MelSpectrogram()
            with torch.no_grad():
                S_dB = mel_extractor(audio)
            
            # Convert to numpy
            S_dB = S_dB.squeeze().numpy()
            
            self.finished.emit(S_dB, sr, None)
        except Exception as e:
            self.finished.emit(None, None, e)

class MelSpectrogramWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMaximumHeight(300)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.updateGeometry()
        self.layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        self.reset_plot()
        self.worker = None

    def reset_plot(self):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, "No Audio Loaded", ha='center', va='center')
        self.canvas.draw()

    def load_audio(self, audio_path):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, "Loading...", ha='center', va='center')
        self.canvas.draw()
        
        if self.worker:
            self.worker.finished.disconnect(self.on_spectrogram_ready)

        self.worker = MelSpectrogramWorker(audio_path)
        self.worker.finished.connect(self.on_spectrogram_ready)
        self.worker.start()

    def on_spectrogram_ready(self, S_dB, sr, error):
        self.ax.clear()
        if error:
            self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, f"Error: {error}", ha='center', va='center')
        else:
            librosa.display.specshow(
                S_dB, x_axis='time', y_axis='mel', sr=sr, 
                hop_length=HOP_LENGTH, fmin=MEL_FMIN, fmax=MEL_FMAX, 
                ax=self.ax, cmap='magma'
            )
            self.ax.set_axis_off()
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas.draw()
