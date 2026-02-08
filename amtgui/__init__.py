
# !/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


import os
#
import torch

from preprocessing.constants import *
#
from . import __path__ as ROOT_PATH

# ##############################################################################
# # PIPELINE STATIC PROPERTIES
# ##############################################################################
WAV_SAMPLERATE = SAMPLE_RATE
MEL_FRAME_SIZE = 2048
MEL_FRAME_HOP = 384
NUM_MELS = N_MELS
NUM_PIANO_KEYS = N_KEYS
MEL_FMIN, MEL_FMAX = (MEL_FMIN, MEL_FMAX)
MEL_WINDOW = torch.hann_window

# ##############################################################################
# # PATHING
# ##############################################################################
ASSETS_PATH = os.path.join(ROOT_PATH[0], "models")
OV_MODEL_PATH = os.path.join(
    ASSETS_PATH,
    "OnsetsAndVelocities_" +
    "2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.pt")
OV_MODEL_CONV1X1_HEAD = [200, 200]
OV_MODEL_LRELU_SLOPE = 0.1

ETE_MODEL_PATH = os.path.join(
    ASSETS_PATH,
    "ete_model_20260128_022829.pt")
ETE_MODEL_INPUT_SHAPE = 1
ETE_MODEL_OUTPUT_SHAPE = NUM_PIANO_KEYS

MODELS = {
    "ov": {
        "path": OV_MODEL_PATH,
        "head": OV_MODEL_CONV1X1_HEAD,
        "slope": OV_MODEL_LRELU_SLOPE
    },
    "endtoend": {
        "path": ETE_MODEL_PATH,
        "input_shape": ETE_MODEL_INPUT_SHAPE,
        "output_shape": ETE_MODEL_OUTPUT_SHAPE
    }
}
DEFAULT_MODEL = "ov"
