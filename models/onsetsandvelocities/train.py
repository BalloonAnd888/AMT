#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import torch
import torch.nn as nn
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from preprocessing.constants import *
from preprocessing.loadDataset import loadDataset
from preprocessing.mel import MelSpectrogram
from models.onsetsandvelocities.ov import OnsetsAndVelocities

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 3

LEARNING_RATE = 1e-4
NUM_MELS = 229
NUM_KEYS = 88
# Model parameters matching the demo configuration
CONV1X1_HEAD = (200, 200)
LRELU_SLOPE = 0.1
IN_CHANS = 2  # Model expects 2 channels (e.g. stereo or duplicated mono)

# Directory to save models: .../onsetsandvelocities/models
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    This module extends ``torch.nn.BCEWithlogitsloss`` with the possibility
    to multiply each scalar loss by a mask number between 0 and 1, before
    aggregating via average.
    """

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs, reduction="none")

    def forward(self, pred, target, mask=None):
        """
        """
        eltwise_loss = super().forward(pred, target)
        if mask is not None:
            assert mask.min() >= 0, "Mask must be in [0, 1]!"
            assert mask.max() <= 1, "Mask must be in [0, 1]!"
            eltwise_loss = eltwise_loss * mask
        result = eltwise_loss.mean()
        #
        return result

def train():
    # 1. Setup
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Models will be saved to: {SAVE_DIR}")

    # 2. Model
    model = OnsetsAndVelocities(
        in_chans=IN_CHANS,
        in_height=NUM_MELS,
        out_height=NUM_KEYS,
        conv1x1head=CONV1X1_HEAD,
        bn_momentum=0.1,
        leaky_relu_slope=LRELU_SLOPE,
        dropout_drop_p=0.2
    ).to(DEVICE)
    
    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_onsets = nn.BCEWithLogitsLoss()
    criterion_vels = MaskedBCEWithLogitsLoss()

    # 4. Data
    dataset = loadDataset(DATA_PATH)

    # Only look at 10 samples for now
    # dataset = Subset(dataset, range(50))

    print(f"Dataset length: {len(dataset)}\n")
    if len(dataset) == 0:
        print("ERROR: The dataset is empty. Please check your data path and file extensions.")
        exit()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Length of dataloader: {len(dataloader)} batches of {BATCH_SIZE}...")

    mel_extractor = MelSpectrogram().to(DEVICE)

    # 5. Training Loop
    for epoch in tqdm(range(NUM_EPOCHS)):
        print(f"\nEpoch {epoch}\n--------")
        train_loss = 0
        train_acc = 0
        model.train()

        for batch_idx, batch in enumerate(dataloader):
            audio = batch['audio'].to(DEVICE)
            onsets = batch['onset'].to(DEVICE).float()
            velocity = batch['velocity'].to(DEVICE).float()

            # Transpose to (Batch, Keys, Time) and slice to match model output (T-1)
            onsets = onsets.permute(0, 2, 1)[:, :, 1:]
            velocity = velocity.permute(0, 2, 1)[:, :, 1:]

            mel = mel_extractor(audio) 
            mel = mel.squeeze(0)

            # Forward pass
            probs_stack, vels = model(mel)

            # Calculate Loss (using the last output of the stack for onsets)
            loss_onset = criterion_onsets(probs_stack[-1], onsets)
            loss_vel = criterion_vels(torch.sigmoid(vels), velocity)
            loss = loss_onset + loss_vel
            train_loss += loss

            # Calculate Accuracy
            pred_onsets = (torch.sigmoid(probs_stack[-1]) > 0.5).float()
            train_acc += (pred_onsets == onsets).float().mean().item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if (batch_idx + 1) % 50 == 0:
                print(f"Looked at {(batch_idx + 1) * BATCH_SIZE}/{len(dataloader.dataset)} samples")

        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
        
    # 6. Save Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ov_model_{timestamp}.pt"))

if __name__ == "__main__":
    train()
