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
NUM_EPOCHS = 10

# Phase Toggle: Set False and provide SNAPSHOT_INPATH to train only velocity
TRAINABLE_ONSETS: bool = True 
SNAPSHOT_INPATH: Optional[str] = None # e.g., "models/ov_model_onsets_only.pt"

CONV1X1_HEAD: List[int] = (200, 200)

# Optimizer Hyperparameters
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

# Loss Constants
ONSET_POSITIVES_WEIGHT: float = 8.0
VEL_LOSS_LAMBDA: float = 10.0 # Weight for velocity loss

IN_CHANS = 2 
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# ##############################################################################
# LOSS FUNCTION
# ##############################################################################
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

# ##############################################################################
# TRAINING FUNCTION
# ##############################################################################
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Models will be saved to: {SAVE_DIR}")
    print(f"Device: {DEVICE} | Training Onsets: {TRAINABLE_ONSETS}")

    # 1. Model Initialization
    model = OnsetsAndVelocities(
        in_chans=IN_CHANS,
        in_height=N_MELS,
        out_height=N_KEYS,
        conv1x1head=CONV1X1_HEAD,
        bn_momentum=BATCH_NORM,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        dropout_drop_p=DROPOUT
    ).to(DEVICE)

    # Load pre-trained onset weights if starting Phase 2
    if SNAPSHOT_INPATH and os.path.exists(SNAPSHOT_INPATH):
        print(f"Loading weights from {SNAPSHOT_INPATH}...")
        model.load_state_dict(torch.load(SNAPSHOT_INPATH, map_location=DEVICE))

    # 2. Optimizer & Loss Setup
    # Filter params: if not training onsets, only optimize the velocity head
    trainable_params = model.parameters() if TRAINABLE_ONSETS else model.velocity_stage.parameters()

    def model_saver(cycle=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ov_checkpoint_{timestamp}.pt"))

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

    # 3. Data Loading
    dataset = loadDataset(DATA_PATH)

    # Only look at 10 samples for now
    dataset = Subset(dataset, range(50))

    print(f"Dataset length: {len(dataset)}\n")
    if len(dataset) == 0:
        print("ERROR: The dataset is empty. Please check your data path and file extensions.")
        exit()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Length of dataloader: {len(dataloader)} batches of {BATCH_SIZE}...")

    mel_extractor = MelSpectrogram().to(DEVICE)

    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_metrics = {k: 0.0 for k in ["loss", "ons_loss", "vel_loss", "ons_acc", "vel_acc"]}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move and Format Data
            audio = batch['audio'].to(DEVICE)
            # Targets: (Batch, Time, Keys) -> (Batch, Keys, Time-1)
            onsets_gt = batch['onset'].to(DEVICE).float().permute(0, 2, 1)[:, :, 1:]
            velocity_gt = batch['velocity'].to(DEVICE).float().permute(0, 2, 1)[:, :, 1:]

            # Preprocessing
            mel = mel_extractor(audio).squeeze(0)
            
            # Forward Pass
            # Passing TRAINABLE_ONSETS helps internal model logic if supported
            probs_stack, vels = model(mel)

            # --- CALCULATE LOSSES ---
            # Onset Loss: Average over the hourglass stacks
            loss_onset = sum(ons_loss_fn(p, (onsets_gt > 0).float()) for p in probs_stack) / len(probs_stack)
            
            # Velocity Loss: Masked by onset ground truth and normalized (0-1)
            vel_mask = (onsets_gt > 0).float()
            vel_target = velocity_gt / 127.0
            loss_vel = vel_loss_fn(vels, vel_target, mask=vel_mask)

            # Combined Loss (Phase Dependent)
            if TRAINABLE_ONSETS:
                loss = loss_onset + (loss_vel * VEL_LOSS_LAMBDA)
            else:
                loss = loss_vel * VEL_LOSS_LAMBDA

            # --- BACKWARD PASS ---
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # --- METRICS (Calculated for Active Notes Only) ---
            with torch.no_grad():
                # Onset Acc
                pred_ons = (torch.sigmoid(probs_stack[-1]) > 0.5).float()
                ons_acc = (pred_ons == (onsets_gt > 0).float()).float().mean()
                
                # Velocity Acc (Mean Absolute Error < 10% on active onsets)
                pred_vels = torch.sigmoid(vels)
                if vel_mask.sum() > 0:
                    active_err = torch.abs(pred_vels[vel_mask > 0] - vel_target[vel_mask > 0])
                    vel_acc = (active_err < 0.1).float().mean()
                else:
                    vel_acc = torch.tensor(0.0)

            # Logging
            total_metrics["loss"] += loss.item()
            total_metrics["ons_loss"] += loss_onset.item()
            total_metrics["vel_loss"] += loss_vel.item()
            total_metrics["ons_acc"] += ons_acc.item()
            total_metrics["vel_acc"] += vel_acc.item()

            if (batch_idx + 1) % 10 == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "On_Acc": f"{ons_acc.item():.2%}"})
        
        # Epoch Summary
        avg = {k: v / len(dataloader) for k, v in total_metrics.items()}
        print(f"\n[Epoch {epoch} Summary]")
        print(f"Train Loss: {avg['loss']:.4f} (Onset: {avg['ons_loss']:.4f}, Vel: {avg['vel_loss']:.4f})")
        print(f"Train Accuracy: Onset {avg['ons_acc']:.2%}, Velocity {avg['vel_acc']:.2%}")

    # Final Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"ov_model_{timestamp}.pt"))

if __name__ == "__main__":
    train()