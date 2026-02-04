import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

from models.endtoend.endtoend import ETE
from models.utils.constants import DEVICE
from preprocessing.constants import DATA_PATH, N_KEYS, SEQUENCE_LENGTH, SAMPLE_RATE, HOP_LENGTH, MEL_FMIN, MEL_FMAX
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram

def midi_to_note_name(midi_number):
    """Converts a MIDI number to a note name (e.g., 60 -> C4)."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"

def inference(model_path):
    print(f"Device: {DEVICE}")
    
    # 1. Load Model
    model = ETE(
        input_shape=1,
        output_shape=N_KEYS
    ).to(DEVICE)

    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    model.eval()
    mel_extractor = MelSpectrogram().to(DEVICE)

    # 2. Load Dataset
    # Using 'test' group
    test_dataset = MAESTRO(path=DATA_PATH, groups=['test'], sequence_length=SEQUENCE_LENGTH)
    
    if len(test_dataset) == 0:
        print("ERROR: Test dataset is empty.")
        return

    # 3. Get Random Sample
    idx = random.randint(0, len(test_dataset) - 1)
    print(f"Visualizing sample index: {idx}")
    
    sample = test_dataset[idx]
    audio = sample['audio'].to(DEVICE) # (Samples)
    onset_target = sample['onset'].to(DEVICE).float() # (Time, Keys)

    # 4. Prepare Input (Similar to train.py)
    # Calculate Mel Spectrogram for the whole audio
    full_mels = mel_extractor(audio.unsqueeze(0)).unsqueeze(1) # (1, 1, F, T)
    
    window_size = full_mels.shape[-1]
    half_window = window_size // 2
    padded_mels = torch.nn.functional.pad(full_mels, (half_window, half_window), mode='constant', value=full_mels.min())

    audio_np = audio.cpu().numpy()
    
    # Detect onsets
    onset_frames = librosa.onset.onset_detect(
        y=audio_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, units='frames'
    )
    
    if len(onset_frames) == 0:
        onset_env = librosa.onset.onset_strength(y=audio_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        onset_frames = [np.argmax(onset_env)]

    print(f"Detected {len(onset_frames)} onsets at frames: {onset_frames}")

    # Prepare batch of windows
    mel_windows = []
    valid_onset_frames = []

    for t in onset_frames:
        if t >= onset_target.shape[0]:
            continue
        
        # Extract window centered at t
        mel_window = padded_mels[:, :, :, t : t + window_size]
        mel_windows.append(mel_window)
        valid_onset_frames.append(t)

    if not mel_windows:
        print("No valid onsets found within bounds.")
        return

    # Stack and predict
    mels_tensor = torch.cat(mel_windows, dim=0) # (Batch, 1, F, T)
    
    print(f"Processing {len(valid_onset_frames)} valid onset frames.")
    with torch.no_grad():
        logits = model(mels_tensor)
        probs = torch.sigmoid(logits) # (Batch, Keys)

    print(f"Model input shape: {mels_tensor.shape}")
    print(f"Logits range: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
    print(f"Probs range: min={probs.min().item():.4f}, max={probs.max().item():.4f}")
    if probs.max() < 0.5:
        print("WARNING: Max probability is below 0.5. No predictions will be shown with threshold 0.5.")

    # 5. Construct Prediction Roll
    pred_roll = np.zeros_like(onset_target.cpu().numpy()) # (Time, Keys)
    probs_np = probs.cpu().numpy()
    
    for i, t in enumerate(valid_onset_frames):
        pred_roll[t, :] = probs_np[i, :]

    # 6. Visualization
    onset_target_np = onset_target.cpu().numpy()
    
    # Print notes for all detected onsets
    print("\n--- Predictions vs Ground Truth ---")
    for i, t in enumerate(valid_onset_frames):
        gt_frame = onset_target_np[t]
        pred_frame = probs_np[i]
        
        threshold = 0.5
        gt_notes = [midi_to_note_name(k + 21) for k in np.where(gt_frame > threshold)[0]]
        pred_notes = [midi_to_note_name(k + 21) for k in np.where(pred_frame > threshold)[0]]
        
        print(f"[Frame {t}] Ground Truth: {gt_notes} | Predicted: {pred_notes}")
    print("-----------------------------------\n")

    mel_np = full_mels.squeeze().cpu().numpy()
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    
    # 1. Mel Spectrogram
    librosa.display.specshow(mel_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=MEL_FMIN, fmax=MEL_FMAX, ax=ax[0], cmap='magma')
    ax[0].set_title('Input: Mel Spectrogram')

    # 2. Target Onsets
    librosa.display.specshow(onset_target_np.T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[1], cmap='Greys', vmin=0, vmax=1)
    ax[1].set_title('Ground Truth Onsets')

    # 3. Predicted Onsets
    librosa.display.specshow(pred_roll.T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[2], cmap='Greys', vmin=0, vmax=1)
    ax[2].set_title('Predicted Onsets (at detected frames)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Attempt to find the latest model in the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")
    
    MODEL_PATH = os.path.join(model_dir, 'ete_model_20260203_195050.pt')

    inference(MODEL_PATH)
