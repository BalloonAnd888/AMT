import librosa.display
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from preprocessing.constants import SAMPLE_RATE, HOP_LENGTH

class PianoRollWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        self.reset_plot()

    def reset_plot(self):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, "No Transcription", ha='center', va='center')
        self.canvas.draw()

    def show_loading(self):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, "Transcribing...", ha='center', va='center')
        self.canvas.draw()

    def plot_pianoroll(self, pianoroll):
        self.ax.clear()
        # Ensure shape is (Keys, Time) for specshow
        if pianoroll.shape[0] > pianoroll.shape[1]:
             pianoroll = pianoroll.T
             
        librosa.display.specshow(
            pianoroll, 
            y_axis='cqt_note', 
            x_axis='time', 
            sr=SAMPLE_RATE, 
            hop_length=HOP_LENGTH, 
            ax=self.ax, 
            cmap='magma',
            fmin=librosa.note_to_hz('A0')
        )
        self.ax.set_axis_off()
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas.draw()
        