
#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
General GUI-related functionality
"""


import io
import datetime
#
from scipy.io import wavfile
from PySide6 import QtCore, QtGui


# #############################################################################
# ## CONVENIENCE FUNCTIONS
# #############################################################################
def change_label_font(lbl, weight=50, size_pt=None):
    """
    """
    fnt = lbl.font()
    if isinstance(weight, int):
        # Map legacy Qt5 weights (0-99) to Qt6 QFont.Weight enum
        if weight < 37:
            w = QtGui.QFont.Weight.Light
        elif weight < 57:
            w = QtGui.QFont.Weight.Normal
        elif weight < 69:
            w = QtGui.QFont.Weight.DemiBold
        elif weight < 81:
            w = QtGui.QFont.Weight.Bold
        else:
            w = QtGui.QFont.Weight.Black
        fnt.setWeight(w)
    else:
        fnt.setWeight(weight)
    if size_pt is not None:
        fnt.setPointSize(size_pt)
    lbl.setFont(fnt)


def resize_button(b, w_ratio=1.0, h_ratio=1.0,
                  padding_px_lrtb=(0, 0, 0, 0)):
    """
    """
    w, h = b.iconSize().toTuple()
    new_w = int(w * w_ratio)
    new_h = int(h * h_ratio)
    new_sz = QtCore.QSize(new_w, new_h)
    b.setIconSize(new_sz)
    #
    left, right, top, bottom = padding_px_lrtb
    b.setStyleSheet(f"padding-left: {left}px;padding-right: {right}px;"
                    f"padding-top: {top}px;padding-bottom: {bottom}px;")


def seconds_to_timestamp(secs, num_decimals=2):
    """
    Given seconds as a float number, returns a string in the form
    ``hh:mm:ss.xx`` where ``xx`` corresponds to the number of decimals
    given at construction.
    """
    secs, fraction = divmod(secs, 1)
    secs_str = str(datetime.timedelta(seconds=int(secs)))
    frac_str = ("{:." + str(num_decimals) + "f}").format(
        round(fraction, num_decimals))
    result = secs_str + frac_str[1:]
    return result


# #############################################################################
# ## AUDIO
# #############################################################################
class QStream(QtCore.QBuffer):
    """
    A stream to make a numpy array playable e.g. in QMediaPlayer, via:

    qstream = QStream(arr, 44100)
    qstream.open()
    QMediaPlayer.SetMedia(QtMultimedia.QMediaContent(), qstream)
    ...
    qstream.close()

    Modified from https://stackoverflow.com/a/63388107/4511978
    """

    READONLY_MODE = QtCore.QIODevice.OpenModeFlag.ReadOnly

    def __init__(self, arr, samplerate, *args, **kwargs):
        """
        :param arr: A float numpy array of rank 1
        :param samplerate: In Hz
        """
        super().__init__(*args, **kwargs)
        # copy the array to bytes
        self.binstream = io.BytesIO()
        wavfile.write(self.binstream, samplerate, arr)
        # copy the bytes to self
        self.setData(self.binstream.getvalue())

    def open(self, mode=None):
        """
        """
        if mode is None:
            mode = self.READONLY_MODE
        super().open(mode)
