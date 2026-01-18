#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing.constants import *
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram
from models.onsetsandvelocities.ov import OnsetsAndVelocities
from models.onsetsandvelocities.decoder import OnsetVelocityNmsDecoder
from models.onsetsandvelocities.visualize import visualize_prediction

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONV1X1_HEAD = (200, 200)
BATCH_NORM = 0.95
LEAKY_RELU_SLOPE = 0.1
DROPOUT = 0.15
IN_CHANS = 2

# Loss Constants
ONSET_POSITIVES_WEIGHT = 8.0
VEL_LOSS_LAMBDA = 10.0

class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    This module extends ``torch.nn.BCEWithlogitsloss`` with the possibility
    to multiply each scalar loss by a mask number between 0 and 1, before
    aggregating via average.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, reduction="none")

    def forward(self, pred, target, mask=None):
        eltwise_loss = super().forward(pred, target)
        if mask is not None:
            assert mask.min() >= 0, "Mask must be in [0, 1]!"
            assert mask.max() <= 1, "Mask must be in [0, 1]!"
            eltwise_loss = eltwise_loss * mask
        result = eltwise_loss.mean()
        return result

def evaluate(model_path, data_path, batch_size):
    print(f"Loading model from: {model_path}")
    print(f"Data path: {data_path}")
    print(f"Device: {DEVICE}")

    # 1. Model Initialization
    # N_MELS and N_KEYS are expected to be in preprocessing.constants
    model = OnsetsAndVelocities(
        in_chans=IN_CHANS,
        in_height=N_MELS,
        out_height=N_KEYS,
        conv1x1head=CONV1X1_HEAD,
        bn_momentum=BATCH_NORM,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        dropout_drop_p=DROPOUT
    ).to(DEVICE)

    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    model.eval()

    # 2. Data Loading
    # Using 'test' group for evaluation
    # SEQUENCE_LENGTH is expected to be in preprocessing.constants
    test_dataset = MAESTRO(path=data_path, groups=['test'], sequence_length=SEQUENCE_LENGTH)
        
    print(f"test_dataset length: {len(test_dataset)}\n")
    if len(test_dataset) == 0:
        print("ERROR: The dataset is empty. Please check your data path and file extensions.")
        exit()

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Batches: {len(test_dataloader)}")

    mel_extractor = MelSpectrogram().to(DEVICE)

    # Loss functions for reporting
    ons_pos_weights = torch.FloatTensor([ONSET_POSITIVES_WEIGHT]).to(DEVICE)
    ons_loss_fn = nn.BCEWithLogitsLoss(pos_weight=ons_pos_weights)
    vel_loss_fn = MaskedBCEWithLogitsLoss()

    # Initialize Decoder
    decoder = OnsetVelocityNmsDecoder(
        num_keys=N_KEYS,
        nms_pool_ksize=3,
        gauss_conv_stddev=1.0,  # Optional: smooths probabilities before NMS
        vel_pad_left=1,
        vel_pad_right=1
    ).to(DEVICE)

    # Metrics
    metrics = {
        "loss": 0.0,
        "ons_loss": 0.0,
        "vel_loss": 0.0,
        "ons_acc": 0.0,
        "vel_acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }

    with torch.inference_mode():
        progress_bar = tqdm(test_dataloader, desc="Testing")
        for batch in progress_bar:
            audio = batch['audio'].to(DEVICE)
            # Targets: (Batch, Time, Keys) -> (Batch, Keys, Time-1)
            onset = batch['onset'].to(DEVICE).float().permute(0, 2, 1)[:, :, 1:]
            velocity = batch['velocity'].to(DEVICE).float().permute(0, 2, 1)[:, :, 1:]

            mel = mel_extractor(audio)

            # Forward Pass
            pred_onset_stack, pred_vels = model(mel, trainable_onsets=False)

            # Calculate loss (for reference)
            batch_ons_loss = sum(ons_loss_fn(p, (onset > 0).float()) for p in pred_onset_stack) / len(pred_onset_stack)
            
            vel_mask = (onset > 0).float()
            vel_target = velocity / 127.0
            batch_vel_loss = vel_loss_fn(pred_vels, vel_target, mask=vel_mask)
            
            loss = batch_ons_loss + (batch_vel_loss * VEL_LOSS_LAMBDA)

            # --- DECODER USAGE ---
            # 1. Apply Sigmoid to get probabilities [0, 1]
            pred_probs = torch.sigmoid(pred_onset_stack[-1])
            pred_vels_probs = torch.sigmoid(pred_vels)
            # 2. Decode to DataFrame (columns: batch_idx, key, t_idx, prob, vel)
            decoded_df = decoder(pred_probs, pred_vels_probs, pthresh=0.5)

            # Calculate metrics
            # Use the last onset stack for predictions
            pred_ons_probs = torch.sigmoid(pred_onset_stack[-1])
            pred_ons_binary = (pred_ons_probs > 0.5).float()
            target_ons_binary = (onset > 0).float()

            # Accuracy
            ons_acc = (pred_ons_binary == target_ons_binary).float().mean()

            pred_vels_sigmoid = torch.sigmoid(pred_vels)
            if vel_mask.sum() > 0:
                active_err = torch.abs(pred_vels_sigmoid[vel_mask > 0] - vel_target[vel_mask > 0])
                vel_acc = (active_err < 0.1).float().mean()
            else:
                vel_acc = torch.tensor(0.0, device=DEVICE)

            # Precision, Recall, F1 (Frame-level)
            tp = (pred_ons_binary * target_ons_binary).sum()
            fp = (pred_ons_binary * (1 - target_ons_binary)).sum()
            fn = ((1 - pred_ons_binary) * target_ons_binary).sum()
            
            eps = 1e-7
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * (precision * recall) / (precision + recall + eps)

            metrics["loss"] += loss.item()
            metrics["ons_loss"] += batch_ons_loss.item()
            metrics["vel_loss"] += batch_vel_loss.item()
            metrics["ons_acc"] += ons_acc.item()
            metrics["vel_acc"] += vel_acc.item()
            metrics["precision"] += precision.item()
            metrics["recall"] += recall.item()
            metrics["f1"] += f1.item()

    # Average metrics
    for k in metrics:
        metrics[k] /= len(test_dataloader)

    print("\n" + "="*40)
    print("TEST RESULTS")
    print("="*40)
    print(f"Total Loss:       {metrics['loss']:.5f}")
    print(f"Onset Loss:       {metrics['ons_loss']:.5f}")
    print(f"Velocity Loss:    {metrics['vel_loss']:.5f}")
    print("-" * 20)
    print(f"Onset Accuracy:   {metrics['ons_acc']:.2%}")
    print(f"Velocity Accuracy:{metrics['vel_acc']:.2%}")
    print("-" * 20)
    print(f"Onset Precision:  {metrics['precision']:.4f}")
    print(f"Onset Recall:     {metrics['recall']:.4f}")
    print(f"Onset F1 Score:   {metrics['f1']:.4f}")
    print("="*40)

    visualize_prediction(model, test_dataset, device=DEVICE)

if __name__ == "__main__":
    # Default to the models directory relative to this script
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    MODEL_PATH = os.path.join(model_dir, 'ov_model_20260118_003304OV.pt')
    evaluate(MODEL_PATH, DATA_PATH, batch_size=8)
    