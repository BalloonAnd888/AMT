import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import librosa
import librosa.display
from torchinfo import summary

from models.onsetsandvelocities.ov import OnsetsAndVelocities
from models.onsetsandvelocities.decoder import OnsetVelocityNmsDecoder
from preprocessing.constants import DATA_PATH, N_KEYS, SEQUENCE_LENGTH, SAMPLE_RATE, HOP_LENGTH, MEL_FMIN, MEL_FMAX, N_MELS
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram
from models.utils.utils import show_mel

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONV1X1_HEAD = (200, 200)
BATCH_NORM = 0.95
LEAKY_RELU_SLOPE = 0.1
DROPOUT = 0.15
IN_CHANS = 2

def inference(model_path, audio_path=None):
    print(f"Device: {DEVICE}")

    # 1. Load Model
    model = OnsetsAndVelocities(
        in_chans=IN_CHANS,
        in_height=N_MELS,
        out_height=N_KEYS,
        conv1x1head=CONV1X1_HEAD,
        bn_momentum=0,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        dropout_drop_p=0
    ).to(DEVICE)

    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    summary(model)
    model.eval()
    mel_extractor = MelSpectrogram().to(DEVICE)

    if audio_path:
        print(f"Inference on audio file: {audio_path}")
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio = torch.tensor(audio).to(DEVICE)
        onset_target = None
        velocity_target = None
        name = os.path.basename(audio_path)
    else:
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
        # Targets are (Time, Keys)
        onset_target = sample['onset'].to(DEVICE).float() 
        velocity_target = sample['velocity'].to(DEVICE).float()
        name = os.path.basename(sample['path']) if 'path' in sample else str(idx)

    # 4. Prepare Input
    # Mel extractor expects (Batch, Samples)
    mel = mel_extractor(audio.unsqueeze(0)) # (1, F, T)
    
    print(f"Audio Duration: {audio.shape[0]/SAMPLE_RATE:.2f}s")
    print(f"Mel Shape: {mel.shape}")

    # Initialize Decoder
    decoder = OnsetVelocityNmsDecoder(
        num_keys=N_KEYS,
        nms_pool_ksize=3,
        gauss_conv_stddev=1,
        gauss_conv_ksize=11,
        vel_pad_left=1,
        vel_pad_right=1
    ).to(DEVICE)

    with torch.inference_mode():
        # Forward Pass
        pred_onset_stack, pred_vels = model(mel, trainable_onsets=False)

        # Pad to restore original length T (since model output is T-1). Keep batch dim for decoder.
        pred_onset_probs = F.pad(torch.sigmoid(pred_onset_stack[-1]), (1, 0))
        pred_vel_probs = F.pad(torch.sigmoid(pred_vels), (1, 0))

        # Decode
        df = decoder(pred_onset_probs, pred_vel_probs, pthresh=0.5)

        # print(f"Pred Onset Probs: {pred_onset_probs}")
        # print(f"Pred Vel Probs: {pred_vel_probs}")
        # print(f"Max Onset Prob: {pred_onset_probs.max().item()}")

    # Sparse matrix reconstruction (Velocities at detected onsets)
    sparse_preds = torch.zeros_like(pred_onset_probs)
    if not df.empty:
        sparse_preds[df["batch_idx"], df["key"], df["t_idx"]] = torch.from_numpy(df["vel"].to_numpy()).to(sparse_preds.device)

    # 5. Visualization Preparation
    mel_np = mel.squeeze(0).cpu().numpy()
    
    # Predictions are (1, Keys, T). Squeeze batch.
    pred_onset_np = pred_onset_probs.squeeze(0).cpu().numpy()
    pred_vel_np = pred_vel_probs.squeeze(0).cpu().numpy()
    sparse_preds_np = sparse_preds.squeeze(0).cpu().numpy()

    print(f"Pred Onset: {pred_onset_np}")
    print(f"Pred Onset Shape: {pred_onset_np.shape}")

    if onset_target is not None:
        # Targets need to be transposed to (Keys, Time) for specshow
        onset_target_np = onset_target.cpu().numpy().T
        velocity_target_np = velocity_target.cpu().numpy().T

    # Calculate extent for imshow
    duration = audio.shape[-1] / SAMPLE_RATE
    extent = [0, duration, 0, N_KEYS]

    # 6. Plot
    if onset_target is not None:
        fig, ax = plt.subplots(6, 1, figsize=(16, 24), sharex=True)
        fig.suptitle(f"Sample: {name}", fontsize=16)

        # 1. Mel
        librosa.display.specshow(mel_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=MEL_FMIN, fmax=MEL_FMAX, ax=ax[0], cmap='magma')
        ax[0].set_title('Input: Mel Spectrogram')

        # 2. GT Onset
        ax[1].imshow(onset_target_np, origin='lower', aspect='auto', interpolation='nearest', cmap='Greys', extent=extent)
        ax[1].set_title('Target: Onsets')
        ax[1].set_ylabel('Key')

        # 3. GT Velocity
        ax[2].imshow(velocity_target_np / 127.0, origin='lower', aspect='auto', interpolation='nearest', cmap='magma', extent=extent)
        ax[2].set_title('Target: Velocity')
        ax[2].set_ylabel('Key')

        # 4. Predicted Onset (raw)
        ax[3].imshow(pred_onset_np, origin='lower', aspect='auto', interpolation='nearest', cmap='Greys', extent=extent)
        ax[3].set_title('Prediction: Onsets (Raw Probs)')
        ax[3].set_ylabel('Key')

        # 5. Predicted Velocity (raw)
        ax[4].imshow(pred_vel_np, origin='lower', aspect='auto', interpolation='nearest', cmap='magma', extent=extent)
        ax[4].set_title('Prediction: Velocity (Raw Probs)')
        ax[4].set_ylabel('Key')

        # 6. Decoded Onsets
        ax[5].imshow(sparse_preds_np, origin='lower', aspect='auto', interpolation='nearest', cmap='magma', extent=extent)
        ax[5].set_title('Prediction: Onsets (Decoded)')
        ax[5].set_xlabel('Time (s)')
        ax[5].set_ylabel('Key')
    else:
        fig, ax = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
        fig.suptitle(f"File: {name}", fontsize=16)

        # 1. Mel
        librosa.display.specshow(mel_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=MEL_FMIN, fmax=MEL_FMAX, ax=ax[0], cmap='magma')
        ax[0].set_title('Input: Mel Spectrogram')

        # 2. Predicted Onset (raw)
        ax[1].imshow(pred_onset_np, origin='lower', aspect='auto', interpolation='nearest', cmap='Greys', extent=extent)
        ax[1].set_title('Prediction: Onsets (Raw Probs)')
        ax[1].set_ylabel('Key')

        # 3. Predicted Velocity (raw)
        ax[2].imshow(pred_vel_np, origin='lower', aspect='auto', interpolation='nearest', cmap='magma', extent=extent)
        ax[2].set_title('Prediction: Velocity (Raw Probs)')
        ax[2].set_ylabel('Key')

        # 4. Decoded Onsets
        ax[3].imshow(sparse_preds_np, origin='lower', aspect='auto', interpolation='nearest', cmap='magma', extent=extent)
        ax[3].set_title('Prediction: Onsets (Decoded)')
        ax[3].set_xlabel('Time (s)')
        ax[3].set_ylabel('Key')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Attempt to find the latest model in the models directory
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # model_dir = os.path.join(current_dir, "models")

    # audio_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'audio'))

    # MODEL_PATH = os.path.join(model_dir, 'OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.pt')
    
    # Find latest model
    # MODEL_PATH = ""
    # if os.path.exists(model_dir):
    #     model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'ov_model' in f]
    #     if model_files:
    #         model_files.sort()
    #         latest_model = model_files[-1]
    #         MODEL_PATH = os.path.join(model_dir, latest_model)
    
    # if not MODEL_PATH:
    #     # Fallback or specific path
    #     MODEL_PATH = os.path.join(model_dir, 'ov_model_latest.pt')

    # print(MODEL_PATH)
    # inference(MODEL_PATH)

    torch.set_printoptions(profile="full")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")
    audio_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'audio'))

    MODEL_PATH = os.path.join(model_dir, 'OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.pt')
    
    # audio_path = os.path.join(audio_dir, 'route1.wav')
    
    # inference(MODEL_PATH, audio_path)
    inference(MODEL_PATH)
