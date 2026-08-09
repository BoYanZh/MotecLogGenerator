"""Pure data models for telemetry channels and messages."""
from __future__ import annotations

import math

import numpy as np

from constants import DISCRETE_CHANNELS
from core.interp import _interp_zoh, _mask_interp_gaps


class Message(object):
    """ A single message in a time series of data. """

    def __init__(self, timestamp: float = 0.0, value: float = 0.0):
        self.timestamp = float(timestamp)
        self.value = float(value)

    def __str__(self):
        return ("t=%f, value=%f" % (self.timestamp, self.value))


class Channel(object):
    """ Represents a singe channel of data containing a time series of values."""
    def __init__(self, name, units, data_type, decimals, messages=None):
        self.name = str(name).strip()
        self.units = str(units)
        self.data_type = data_type
        self.decimals = decimals
        if messages:
            self.messages = messages
        else:
            self.messages = []

    def start(self):
        return self.messages[0].timestamp if self.messages else math.inf

    def end(self):
        return self.messages[-1].timestamp if self.messages else -math.inf

    def avg_frequency(self):
        if len(self.messages) >= 2:
            dt = self.end() - self.start()
            return len(self.messages) / dt
        else:
            return 0

    def resample(self, start_time, end_time, frequency, mask_interp_gaps=False):
        if not self.messages:
            return

        num_msgs = math.floor(frequency * (end_time - start_time))
        if num_msgs < 1:
            return
        dt_step = 1.0 / frequency

        src_t = np.array([m.timestamp for m in self.messages])
        src_v = np.array([m.value for m in self.messages])
        new_t = start_time + dt_step * np.arange(num_msgs)

        if self.name in DISCRETE_CHANNELS:
            new_v = _interp_zoh(new_t, src_t, src_v)
        else:
            new_v = np.interp(new_t, src_t, src_v)
            if mask_interp_gaps:
                new_v = _mask_interp_gaps(new_v, new_t, src_t)

        self.messages = [Message(float(new_t[i]), float(new_v[i])) for i in range(num_msgs)]

    def __str__(self):
        s = "Channel: %s, Units: %s, Decimals: %d, Messages: %d, Frequency: %.2f Hz"
        return s % (self.name, self.units, self.decimals, len(self.messages), self.avg_frequency())
