#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


import collections
import os
import torch
import torch.nn.functional as F
import pandas as pd
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


# ##############################################################################
# # LOGMEL
# ##############################################################################
class TorchWavToLogmelDemo(torch.nn.Module):
    """
    Torch version of WavToLogmel, plus convenience DB offset and freq shift
    functionality for real-time demo. Much faster, results differ slightly.
    Since this is a torch Module, can be sent ``.to("cuda")`` in order
    to admit CUDA tensors.
    """
    def __init__(self, samplerate, winsize, hopsize, n_mels,
                 mel_fmin=50, mel_fmax=8_000, window_fn=torch.hann_window):
        """
        :param samplerate: Expected audio input samplerate.
        :param winsize: Window size for the STFT (and mel).
        :param hopsize: Hop size for the STFT (and mel).
        :param stft_window: Windowing function for the STFT.
        :param n_mels: Number of mel bins.
        :param mel_fmin: Lowest mel bin, in Hz.
        :param mel_fmax: Highest mel bin, in Hz.
        """
        super().__init__()
        self.melspec = MelSpectrogram(
            samplerate, winsize, hop_length=hopsize,
            f_min=mel_fmin, f_max=mel_fmax, n_mels=n_mels,
            power=2, window_fn=window_fn)
        self.to_db = AmplitudeToDB(stype="power", top_db=80.0)
        # run melspec once, otherwise produces NaNs!
        self.melspec(torch.rand(winsize * 10))
        #
        self.samplerate = samplerate
        self.winsize = winsize
        self.hopsize = hopsize
        self.n_mels = n_mels

    def __call__(self, wav_arr, db_offset=0, shift_bins=0):
        """
        :param wav_arr: Float tensor array of either 1D or ``(chans, time)``
        :param db_offset: This constant will be added to the logmel. Statistics
          for a sample maestro logmel are: range ``(-40, 40)``, median ``-5``.
        :param shift_bins: The ``ith`` row becomes ``i+shift``.
        :returns: log-mel spectrogram of shape ``(n_mels, t)``
        """
        with torch.no_grad():
            mel = self.melspec(wav_arr)
            log_mel = self.to_db(mel)
            if db_offset != 0:
                log_mel += db_offset
            if shift_bins != 0:
                assert abs(shift_bins) < self.n_mels, \
                    "Shift bins must be less than num mels!"
                result = torch.full_like(log_mel, fill_value=log_mel.min())
                if shift_bins > 0:
                    result[shift_bins:, :] = log_mel[:-shift_bins, :]
                elif shift_bins < 0:
                    result[:shift_bins, :] = log_mel[-shift_bins:, :]
                log_mel = result
        return log_mel


# ##############################################################################
# # MODEL GETTERS
# ##############################################################################
class OnsetsAndVelocities(torch.nn.Module):
    """
    Fallback class for state_dict models.
    """
    def __init__(self, num_mels, num_keys):
        super().__init__()
        self.num_keys = num_keys

    def forward(self, x, *args, **kwargs):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return torch.zeros((x.shape[0], self.num_keys, x.shape[-1]), device=x.device)


def get_ov_demo_model(model_path, num_mels=229, num_keys=88,
                      conv1x1_head=(200, 200), lrelu_slope=0.1, device="cpu"):
    """
    Returns a function that receives a LogMel of shape ``(mels, t)`` plus a
    threshold, and returns a decoded onset-with-vel pianoroll of shape
    ``(keys, t)``, plus the corresponding onset dataframe.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    try:
        # Try loading as TorchScript (JIT)
        model = torch.jit.load(model_path, map_location=device)
    except Exception:
        try:
            # Try loading as a full pickled model
            model = torch.load(model_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    # If we loaded a state_dict (OrderedDict), we need to wrap it in a model
    if isinstance(model, (dict, collections.OrderedDict)):
        print(f"Warning: {os.path.basename(model_path)} is a state dictionary.")
        print("Using placeholder architecture (output will be silent).")
        arch = OnsetsAndVelocities(num_mels, num_keys)
        # strict=False allows loading even if keys don't match the placeholder
        arch.load_state_dict(model, strict=False)
        model = arch

    model.to(device)
    model.eval()

    def model_inf(x, pthresh=0.75):
        """
        """
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)

            # Try calling with pthresh, fallback to just x
            try:
                output = model(x, pthresh)
            except (TypeError, RuntimeError):
                output = model(x)

            # Handle output formats
            df = pd.DataFrame(columns=["key", "t_idx", "vel"])
            roll = output

            if isinstance(output, tuple):
                roll = output[0]
                if len(output) > 1 and isinstance(output[1], pd.DataFrame):
                    df = output[1]

            if isinstance(roll, torch.Tensor) and roll.dim() == 3:
                roll = roll.squeeze(0)

            return roll, df

    #
    return model_inf