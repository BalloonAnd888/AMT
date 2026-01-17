#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from typing import List, Optional
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.onsetsandvelocities.optimizers import AdamWR
from preprocessing.constants import *
from preprocessing.loadDataset import loadDataset
from preprocessing.mel import MelSpectrogram
from models.onsetsandvelocities.ov import OnsetsAndVelocities

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 1

CONV1X1_HEAD: List[int] = (200, 200)

# optimizer
LR_MAX: float = 0.008
LR_WARMUP: float = 0.5
LR_PERIOD: int = 1000
LR_DECAY: float = 0.975
LR_SLOWDOWN: float = 1.0
MOMENTUM: float = 0.95
WEIGHT_DECAY: float = 0.0003
BATCH_NORM: float = 0.95
DROPOUT: float = 0.15
LEAKY_RELU_SLOPE: Optional[float] = 0.1

# loss
ONSET_POSITIVES_WEIGHT: float = 8.0
TRAINABLE_ONSETS: bool = True

IN_CHANS = 2  # Model expects 2 channels (e.g. stereo or duplicated mono)

# Directory to save models: .../onsetsandvelocities/models
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

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
        in_height=N_MELS,
        out_height=N_KEYS,
        conv1x1head=CONV1X1_HEAD,
        bn_momentum=BATCH_NORM,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        dropout_drop_p=DROPOUT
    ).to(DEVICE)
    
    # 3. Optimizer & Loss
    trainable_params = model.parameters() if TRAINABLE_ONSETS else \
        model.velocity_stage.parameters()

    def model_saver(cycle=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ov_model_{timestamp}.pt"))

    opt_hpars = {
        "lr_max": LR_MAX, "lr": LR_MAX,
        "lr_period": LR_PERIOD, "lr_decay": LR_DECAY,
        "lr_slowdown": LR_SLOWDOWN, "cycle_end_hook_fn": model_saver,
        "cycle_warmup": LR_WARMUP, "weight_decay": WEIGHT_DECAY,
        "betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False}

    optimizer = AdamWR(trainable_params, **opt_hpars)
    ons_pos_weights = torch.FloatTensor([ONSET_POSITIVES_WEIGHT]).to(DEVICE)
    ons_loss_fn = nn.BCEWithLogitsLoss(pos_weight=ons_pos_weights)
    vel_loss_fn = MaskedBCEWithLogitsLoss()

    # 4. Data
    dataset = loadDataset(DATA_PATH)

    # Only look at 10 samples for now
    # dataset = Subset(dataset, range(300))

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
        train_loss_onset = 0
        train_loss_vel = 0
        train_acc_onset = 0
        train_acc_vel = 0
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
            loss_onset = ons_loss_fn(probs_stack[-1], onsets)
            loss_vel = vel_loss_fn(torch.sigmoid(vels), velocity)
            loss = loss_onset + loss_vel
            
            train_loss += loss.item()
            train_loss_onset += loss_onset.item()
            train_loss_vel += loss_vel.item()

            # Calculate Accuracy
            pred_onsets = (torch.sigmoid(probs_stack[-1]) > 0.5).float()
            train_acc_onset += (pred_onsets == onsets).float().mean().item()

            # Calculate Velocity Accuracy (within 0.1 tolerance)
            train_acc_vel += (torch.abs(torch.sigmoid(vels) - velocity) < 0.1).float().mean().item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if (batch_idx + 1) % 50 == 0:
                print(f"Looked at {(batch_idx + 1) * BATCH_SIZE}/{len(dataloader.dataset)} samples")

        train_loss /= len(dataloader)
        train_loss_onset /= len(dataloader)
        train_loss_vel /= len(dataloader)
        train_acc_onset /= len(dataloader)
        train_acc_vel /= len(dataloader)
        print(f"Train loss: {train_loss:.5f} (Onset: {train_loss_onset:.5f}, Vel: {train_loss_vel:.5f}) | "
              f"Train accuracy: Onset {train_acc_onset*100:.2f}%, Vel {train_acc_vel*100:.2f}%")
        
    # 6. Save Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ov_model_{timestamp}.pt"))

if __name__ == "__main__":
    train()
