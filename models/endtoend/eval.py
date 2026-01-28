import torch
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from models.endtoend.endtoend import ETE
from models.utils.constants import DEVICE
from preprocessing.constants import DATA_PATH, N_KEYS, SEQUENCE_LENGTH
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram

POS_WEIGHT = 10.0

def evaluate(model_path, data_path, batch_size):
    print(f"Loading model from: {model_path}")
    print(f"Data path: {data_path}")
    print(f"Device: {DEVICE}")

    model = ETE(
        input_shape=1,
        output_shape=N_KEYS
    ).to(DEVICE)

    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    # 2. Data Loading
    test_dataset = MAESTRO(path=data_path, groups=['test'], sequence_length=SEQUENCE_LENGTH)
        
    print(f"test_dataset length: {len(test_dataset)}\n")
    if len(test_dataset) == 0:
        print("ERROR: The dataset is empty. Please check your data path and file extensions.")
        exit()

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Batches: {len(test_dataloader)}")

    mel_extractor = MelSpectrogram().to(DEVICE)

    onset_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([POS_WEIGHT]).to(DEVICE))

    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.inference_mode():
        progress_bar = tqdm(test_dataloader, desc="Testing")
        for batch in progress_bar:
            audio = batch['audio'].to(DEVICE)
            onset = batch['onset'].to(DEVICE).float()

            mel = mel_extractor(audio).unsqueeze(1)

            test_onset_pred = model(mel)

            # Compare keys based on the time (using the center frame of the window)
            onset = onset[:, onset.size(1) // 2, :]

            loss = onset_loss_fn(test_onset_pred, onset)
            pred_binary = (torch.sigmoid(test_onset_pred) > 0.5).float()
            acc = (pred_binary == onset).float().mean()

            test_loss += loss.item()
            test_acc += acc.item()

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    print(f"\nTest Loss: {test_loss:.5f} | Test Acc: {test_acc:.2%}")

if __name__ == "__main__":
    # Default to the models directory relative to this script
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    MODEL_PATH = os.path.join(model_dir, 'ov_model_20260126_231149.pt')
    evaluate(MODEL_PATH, DATA_PATH, batch_size=8)
