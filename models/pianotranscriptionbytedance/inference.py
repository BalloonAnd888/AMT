import os
import torch 
import librosa
import urllib.request
from piano_transcription_inference import PianoTranscription, sample_rate

def inference(audio_path):
    # Load audio
    print(audio_path)
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Checkpoint path
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'note_F1=0.9677_pedal_F1=0.9186.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Downloading checkpoint to {checkpoint_path}")
        urllib.request.urlretrieve('https://zenodo.org/record/4034264/files/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth', checkpoint_path)

    # Transcriptor
    transcriptor = PianoTranscription(device='cuda', checkpoint_path=checkpoint_path)    # 'cuda' | 'cpu'

    # Transcribe and write out to MIDI file
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    midi_path = os.path.join(results_dir, os.path.splitext(os.path.basename(audio_path))[0] + '.mid')
    transcribed_dict = transcriptor.transcribe(audio, midi_path)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'audio'))

    audio_path = os.path.join(audio_dir, 'route1.wav')

    inference(audio_path)
    