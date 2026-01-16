#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from preprocessing.dataset import MAESTRO
from preprocessing.constants import *

# Attempt to import the model class.
from models.onsetsandvelocities.ov import OnsetsAndVelocities

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_MELS = 229
NUM_KEYS = 88
# Model parameters matching the demo configuration
CONV1X1_HEAD = (200, 200)
LRELU_SLOPE = 0.1
IN_CHANS = 2  # Model expects 2 channels (e.g. stereo or duplicated mono)

# Directory to save models: .../onsetsandvelocities/models
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

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
    
    # model.train()

    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_onsets = nn.BCEWithLogitsLoss()
    criterion_vels = nn.MSELoss()

    # 4. Data
    dataset = MAESTRO(path=DATA_PATH, groups=[], sequence_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Training Loop
    for epoch in tqdm(range(NUM_EPOCHS)):
        print(f"Epoch {epoch}\n--------")
        train_loss = 0
        for batch_idx, (x, y_onsets, y_vels) in enumerate(dataloader):
            x, y_onsets, y_vels = x.to(DEVICE), y_onsets.to(DEVICE), y_vels.to(DEVICE)
            model.train()

            # Forward pass
            probs_stack, vels = model(x, trainable_onsets=True)
            
            # Calculate Loss (using the last output of the stack for onsets)
            loss_onset = criterion_onsets(probs_stack[-1], y_onsets)
            loss_vel = criterion_vels(torch.sigmoid(vels), y_vels)
            loss = loss_onset + loss_vel

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 400 == 0:
                print(f"Looked at {batch_idx * len(x)}/{len(dataloader.dataset)} samples")

        print(f"Loss: {train_loss / len(dataloader):.4f}")
        
        # 6. Save Model
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ov_model_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    train()
