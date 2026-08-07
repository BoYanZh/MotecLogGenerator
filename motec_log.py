import datetime
import numpy as np
import struct
from data_log import DataLog, Message, Channel
from ldparser.ldparser import ldVehicle, ldVenue, ldEvent, ldHead, ldChan, ldData

class MotecLog(object):
    """ Handles generating a MoTeC .ld file from log data.

    This utilizes the ldparser library for packing all the meta data and channel data into the
    correct binary format. Some functionality and information (e.g. pointer constants below) was
    missing from the ldparser library, so this class servers as a wrapper to fill in the gaps.

    This operates on containers from the data_log library.
    """
    # Pointers to locations in the file where data sections should be written. These have been
    # determined from inspecting some MoTeC .ld files, and were consistent across all files.
    VEHICLE_PTR = 1762
    VENUE_PTR = 5078
    EVENT_PTR = 8180
    HEADER_PTR = 11336

    CHANNEL_HEADER_SIZE = struct.calcsize(ldChan.fmt)

    def __init__(self):
        self.driver = ""
        self.vehicle_id = ""
        self.vehicle_weight = 0
        self.vehicle_type = ""
        self.vehicle_comment = ""
        self.venue_name = ""
        self.event_name = ""
        self.event_session = ""
        self.long_comment = ""
        self.short_comment = ""
        self.datetime = datetime.datetime.now()

        # File components from ldparser
        self.ld_header = None
        self.ld_channels = []

    def initialize(self):
        """ Initializes all the meta data for the motec log.

        This must be called before adding any channel data.
        """
        ld_vehicle = ldVehicle(self.vehicle_id, self.vehicle_weight, self.vehicle_type, \
            self.vehicle_comment)
        ld_venue = ldVenue(self.venue_name, self.VEHICLE_PTR, ld_vehicle)
        ld_event = ldEvent(self.event_name, self.event_session, self.long_comment, \
            self.VENUE_PTR, ld_venue)

        self.ld_header = ldHead(self.HEADER_PTR, self.HEADER_PTR, self.EVENT_PTR, ld_event, \
            self.driver, self.vehicle_id, self.venue_name, self.datetime, self.short_comment, \
            self.event_name, self.event_session)

    def add_channel(self, log_channel):
        """ Adds a single channel of data to the motec log.

        log_channel: data_log.Channel
        """
        data_len = len(log_channel.messages)
        data_type = np.float32 if log_channel.data_type is float else np.int32
        freq = int(round(log_channel.avg_frequency()))
        shift = 0
        multiplier = 1
        scale = 1
        decimals = 0

        ld_channel = ldChan(None, 0, 0, 0, 0, data_len, \
            data_type, freq, shift, multiplier, scale, decimals, log_channel.name, "", \
            log_channel.units)

        # Add in the channel data
        ld_channel._data = np.array([data_type(msg.value) for msg in log_channel.messages], dtype=data_type)
        self.ld_channels.append(ld_channel)

    def add_all_channels(self, data_log):
        """ Adds all channels from a DataLog to the motec log.

        data_log: data_log.DataLog
        """
        for channel_name, channel in data_log.channels.items():
            self.add_channel(channel)

    def _finalize_pointers(self):
        """ Calculates channel header pointers and data offsets using 124B CHANNEL_HEADER_SIZE steps. """
        n = len(self.ld_channels)
        if n == 0:
            return

        meta_base = self.HEADER_PTR
        for i, ch in enumerate(self.ld_channels):
            ch.meta_ptr = meta_base + i * self.CHANNEL_HEADER_SIZE
            ch.prev_meta_ptr = meta_base + (i - 1) * self.CHANNEL_HEADER_SIZE if i > 0 else 0
            ch.next_meta_ptr = meta_base + (i + 1) * self.CHANNEL_HEADER_SIZE if i < n - 1 else 0

        data_ptr = meta_base + n * self.CHANNEL_HEADER_SIZE
        self.ld_header.data_ptr = data_ptr
        for ch in self.ld_channels:
            ch.data_ptr = data_ptr
            data_ptr += ch._data.nbytes

    def write(self, filename):
        """ Writes the motec log data to disc. """
        # Check for the presence of any channels, since the ldData write() method doesn't
        # gracefully handle zero channels
        if self.ld_channels:
            self._finalize_pointers()
            ld_data = ldData(self.ld_header, self.ld_channels)

            # Need to zero out the final channel pointer
            ld_data.channs[-1].next_meta_ptr = 0

            ld_data.write(filename)
        else:
            with open(filename, "wb") as f:
                self.ld_header.write(f, 0)
