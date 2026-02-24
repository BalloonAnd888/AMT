import torch
import librosa
import numpy as np
from PySide6.QtCore import QThread, Signal
from preprocessing.mel import MelSpectrogram
from preprocessing.constants import DEVICE, SAMPLE_RATE

class Transcription(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, model, model_type, audio_path):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.audio_path = audio_path

    def run(self):
        try:
            # Load audio
            audio, _ = librosa.load(self.audio_path, sr=SAMPLE_RATE)
            audio = torch.tensor(audio).to(DEVICE)
            
            mel_extractor = MelSpectrogram().to(DEVICE)
            mel = mel_extractor(audio.unsqueeze(0))

            self.model.eval()
            
            pianoroll = None

            # with torch.no_grad():
            if self.model_type == "Onsets and Frames":
                print("Onsets and Frames")
                onset_pred, offset_pred, _, frame_pred, velocity_pred = self.model(mel)
                pianoroll = frame_pred.squeeze(0).cpu().detach().numpy()

            elif self.model_type == "Onsets and Velocities":
                print("Onsets and Velocities")
                print(self.model)

            elif self.model_type == "End to End":
                print("End to End")
                print(self.model)

            if pianoroll is not None:
                self.finished.emit(pianoroll)
            else:
                self.error.emit("Model returned no prediction.")

        except Exception as e:
            self.error.emit(str(e))
