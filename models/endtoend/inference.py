import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import torch.nn.functional as F

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

def predict_ensemble(model, mel_extractor, audio, device):
    """
    Implements the voting strategy from the paper:
    11 windows shifted by 16.5ms intervals (±82.5ms range).
    Returns the fraction of votes for each key (0.0 to 1.0).
    """
    # Constants from the paper
    SHIFT_MS = 16.5
    STEPS = 5
    
    shift_samples = int((SHIFT_MS / 1000.0) * SAMPLE_RATE)
    original_length = audio.shape[-1]
    
    # Pad audio to allow shifting without changing window size
    max_shift = shift_samples * STEPS
    padded_audio = F.pad(audio, (max_shift, max_shift), mode='constant', value=0)
    
    predictions = []
    
    for i in range(-STEPS, STEPS + 1):
        # Extract shifted window. 
        # i < 0 shifts window left (earlier context), i > 0 shifts right.
        start_idx = max_shift + (i * shift_samples)
        end_idx = start_idx + original_length
        
        audio_window = padded_audio[:, start_idx:end_idx]
        
        mel = mel_extractor(audio_window).unsqueeze(1)
        logits = model(mel)
        probs = torch.sigmoid(logits)
        
        # Binarize prediction for hard voting (as per paper)
        predictions.append((probs > 0.5).float())
    
    # Stack and calculate vote fraction: (11, Keys) -> (Keys)
    predictions = torch.stack(predictions).squeeze(1)
    vote_fraction = predictions.mean(dim=0)
    return vote_fraction

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
    audio = sample['audio'].to(DEVICE).unsqueeze(0) # (1, Samples)
    onset_target = sample['onset'].to(DEVICE).float() # (Time, Keys)

    model.eval()
    # 4. Inference
    with torch.no_grad():
        # Extract Mel Spectrogram
        mel = mel_extractor(audio).unsqueeze(1) # (1, 1, F, T)
        
        # Model Prediction
        onset_logits = model(mel) # (1, Keys)
        onset_probs = torch.sigmoid(onset_logits).squeeze(0) # (Keys)

        # Ensemble Prediction (Paper Method)
        ensemble_probs = predict_ensemble(model, mel_extractor, audio, DEVICE)

    # Process Target for Visualization
    # We want to visualize the whole sequence for context
    onset_target_np = onset_target.cpu().numpy().T # (Keys, Time)

    # 5. Plotting
    onset_probs_np = onset_probs.cpu().numpy()
    
    # Get ground truth for center frame
    center_idx = onset_target_np.shape[1] // 2
    target_frame = onset_target_np[:, center_idx]

    # Convert indices to note names (assuming index 0 = MIDI 21 / A0)
    threshold = 0.5
    active_target_indices = np.where(target_frame > threshold)[0]
    active_pred_indices = np.where(onset_probs_np > threshold)[0]
    active_ensemble_indices = np.where(ensemble_probs.cpu().numpy() > threshold)[0]

    target_notes = [midi_to_note_name(idx + 21) for idx in active_target_indices]
    pred_notes = [midi_to_note_name(idx + 21) for idx in active_pred_indices]
    ensemble_notes = [midi_to_note_name(idx + 21) for idx in active_ensemble_indices]

    print(f"\n[Frame {center_idx}] Ground Truth: {target_notes}")
    print(f"[Frame {center_idx}] Predicted (Single): {pred_notes}")
    print(f"[Frame {center_idx}] Predicted (Vote):   {ensemble_notes}\n")

    mel_np = mel.squeeze().cpu().numpy()
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    
    # 1. Mel Spectrogram
    librosa.display.specshow(mel_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=MEL_FMIN, fmax=MEL_FMAX, ax=ax[0], cmap='magma')
    ax[0].set_title('Input: Mel Spectrogram')

    duration = audio.shape[-1] / SAMPLE_RATE
    extent = [0, duration, 0, N_KEYS]
    
    # 2. Target Onsets
    ax[1].imshow(onset_target_np, origin='lower', aspect='auto', interpolation='nearest', cmap='Greys', extent=extent, vmin=0, vmax=1)
    ax[1].set_title('Target: Onsets')
    ax[1].set_ylabel('Key')
    ax[1].set_xlabel('Time (s)')
    
    # Add marker for center frame
    center_time = (center_idx * HOP_LENGTH) / SAMPLE_RATE
    ax[1].axvline(x=center_time, color='r', linestyle='--', alpha=0.8, label='Prediction Frame')
    ax[1].legend(loc='upper right')

    # 3. Prediction (Bar Graph)
    keys = np.arange(N_KEYS)
    ax[2].bar(keys, onset_probs_np, alpha=0.5, label='Single Prob', color='blue')
    ax[2].bar(keys, ensemble_probs.cpu().numpy(), alpha=0.5, label='Ensemble Vote', color='green')
    
    # Overlay Ground Truth
    active_keys = np.where(target_frame > 0.5)[0]
    ax[2].scatter(active_keys, target_frame[active_keys], color='red', marker='x', s=50, label='Ground Truth', zorder=5)

    ax[2].set_title('Prediction vs Target (Center Frame)')
    ax[2].set_ylabel('Probability')
    ax[2].set_xlabel('MIDI Key Index')
    ax[2].set_ylim(0, 1.1)
    ax[2].legend(loc='upper right')
    ax[2].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Attempt to find the latest model in the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")
    
    MODEL_PATH = os.path.join(model_dir, 'ov_model_20260126_231149.pt')

    inference(MODEL_PATH)
