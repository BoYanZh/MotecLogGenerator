from __future__ import annotations

import datetime
import struct

import numpy as np

from data_log import Channel, DataLog, Message
from ldparser.ldparser import ldChan, ldData, ldEvent, ldHead, ldVehicle, ldVenue


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

    ld_channels: list[ldChan]

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
            self.driver, self.vehicle_id, self.venue_name, self.datetime, self.short_comment)

    def add_channel(self, log_channel: Channel):
        """ Adds a single channel of data to the motec log.

        log_channel: data_log.Channel
        """
        # Advance the header data pointer
        self.ld_header.data_ptr += self.CHANNEL_HEADER_SIZE

        # Advance the data pointers of all previous channels
        for ld_channel in self.ld_channels:
            ld_channel.data_ptr += self.CHANNEL_HEADER_SIZE

        # Determine our file pointers
        if self.ld_channels:
            meta_ptr = self.ld_channels[-1].next_meta_ptr
            prev_meta_ptr = self.ld_channels[-1].meta_ptr
            data_ptr = self.ld_channels[-1].data_ptr + self.ld_channels[-1]._data.nbytes
        else:
            # First channel needs the previous pointer zero'd out
            meta_ptr = self.HEADER_PTR
            prev_meta_ptr = 0
            data_ptr = self.ld_header.data_ptr
        next_meta_ptr = meta_ptr + self.CHANNEL_HEADER_SIZE

        # Channel specs
        data_len = len(log_channel.messages)
        data_type = np.float32 if log_channel.data_type is float else np.int32
        freq = int(round(log_channel.avg_frequency()))
        shift = 0
        multiplier = 1
        scale = 1

        # Decimal places must be hard coded to zero for float32 channels in MoTeC format
        decimals = 0

        ld_channel = ldChan(None, meta_ptr, prev_meta_ptr, next_meta_ptr, data_ptr, data_len, \
            data_type, freq, shift, multiplier, scale, decimals, log_channel.name, "", \
            log_channel.units)

        # Add in the channel data
        ld_channel._data = np.array([], data_type)
        for msg in log_channel.messages:
            ld_channel._data = np.append(ld_channel._data, data_type(msg.value))

        # Add the ld channel and advance the file pointers
        self.ld_channels.append(ld_channel)

    def add_all_channels(self, data_log):
        """ Adds all channels from a DataLog to the motec log.

        data_log: data_log.DataLog
        """
        for channel_name, channel in data_log.channels.items():
            self.add_channel(channel)

    def write(self, filename):
        """ Writes the motec log data to disc. """
        # Check for the presence of any channels, since the ldData write() method doesn't
        # gracefully handle zero channels
        if self.ld_channels:
            ld_data = ldData(self.ld_header, self.ld_channels)

            # Need to zero out the final channel pointer
            ld_data.channs[-1].next_meta_ptr = 0

            ld_data.write(filename)
        else:
            with open(filename, "wb") as f:
                self.ld_header.write(f, 0)

    def write_ldx(self, ldx_filename, laps_info=None):
        """ Writes the associated MoTeC .ldx index file containing lap markers and metadata details. """
        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        root = ET.Element("LDXFile", {
            "Locale": "English_United States.1252",
            "DefaultLocale": "C",
            "Version": "1.6"
        })
        layers = ET.SubElement(root, "Layers")
        layer = ET.SubElement(layers, "Layer")

        marker_block = ET.SubElement(layer, "MarkerBlock")
        marker_group = ET.SubElement(marker_block, "MarkerGroup", {"Name": "Beacons", "Index": "3"})

        laps = laps_info.get("laps", []) if laps_info and "laps" in laps_info else []
        # MoTeC i2 automatically creates Lap 1 from 0.0s to the first Beacon Marker.
        # To avoid 0s or 3s ghost Out Laps:
        # - Never write a beacon at 0.0s.
        # - Write beacons ONLY at lap transition boundaries (end of each lap except the final stint end).
        # - For a single-lap file, no intermediate beacons are written so MoTeC treats 0.0s..end as Lap 1.
        beacon_times = []
        if len(laps) > 1:
            for lap in laps[:-1]:
                end_t = lap.get("end_time", 0.0)
                if end_t > 1.0:
                    beacon_times.append(end_t)

        for idx, b_time in enumerate(beacon_times):
            t_us = b_time * 1e6
            marker_name = f"Manual.{idx + 1}"
            ET.SubElement(marker_group, "Marker", {
                "Version": "100",
                "ClassName": "BCN",
                "Name": marker_name,
                "Flags": "77",
                "Time": f"{t_us:.17e}"
            })

        ET.SubElement(layer, "RangeBlock")

        details = ET.SubElement(layers, "Details")

        def add_detail_str(string_id, val):
            ET.SubElement(details, "String", {"Id": string_id, "Value": str(val) if val is not None else ""})

        add_detail_str("Event", self.event_name)
        add_detail_str("Venue", self.venue_name)
        add_detail_str("Venue Category", "")
        add_detail_str("Driver", self.driver)
        add_detail_str("Team", "")
        add_detail_str("Vehicle Id", self.vehicle_id)
        add_detail_str("Vehicle Number", "")
        add_detail_str("Vehicle Desc", self.vehicle_comment)
        add_detail_str("Engine Id", "")
        add_detail_str("Session", self.event_session)
        add_detail_str("Start Lap", "")
        add_detail_str("Short Comment", self.short_comment)
        add_detail_str("Long Comment", self.long_comment)

        date_str = self.datetime.strftime("%d/%m/%Y")
        time_str = self.datetime.strftime("%H:%M:%S")
        ET.SubElement(details, "DateTime", {"Id": "Log Date", "Value": date_str})
        ET.SubElement(details, "DateTime", {"Id": "Log Time", "Value": time_str})

        add_detail_str("Sky", "")
        add_detail_str("Wind Direction", "")
        add_detail_str("Weather Comment", "")
        add_detail_str("Vehicle Type", self.vehicle_type)
        add_detail_str("Vehicle Drive Type", "")
        add_detail_str("Vehicle Comment", self.vehicle_comment)

        total_laps = laps_info.get("total_laps", 1) if laps_info else 1
        fastest_lap = laps_info.get("fastest_lap", 0) if laps_info else 0
        fastest_time_sec = laps_info.get("fastest_time", 0.0) if laps_info else 0.0

        if fastest_time_sec > 0:
            m = int(fastest_time_sec // 60)
            s = fastest_time_sec % 60
            fastest_time_str = f"{m}:{s:06.3f}"
        else:
            fastest_time_str = "0:00.000"

        add_detail_str("Total Laps", str(total_laps))
        add_detail_str("Fastest Time", fastest_time_str)
        add_detail_str("Fastest Lap", str(fastest_lap))

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" ")
        with open(ldx_filename, "w", encoding="utf-8") as f:
            f.write(xml_str)

    def _find_gps_channels(self, data_log):
        lat_names = ["GPS Latitude", "Latitude", "gps_lat", "lat"]
        lon_names = ["GPS Longitude", "Longitude", "gps_lon", "lon"]
        lat_chan = next((data_log.channels[n] for n in lat_names if n in data_log.channels), None)
        lon_chan = next((data_log.channels[n] for n in lon_names if n in data_log.channels), None)
        return lat_chan, lon_chan

    def write_gpx(self, gpx_filename, data_log):
        """ Export GPS track points as a standard GPX file. """
        lat_chan, lon_chan = self._find_gps_channels(data_log)
        if not lat_chan or not lon_chan:
            return False

        alt_chan = data_log.channels.get("GPS Altitude")
        spd_chan = data_log.channels.get("Ground Speed")

        n = min(len(lat_chan.messages), len(lon_chan.messages))
        if n == 0:
            return False

        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        root = ET.Element("gpx", {
            "version": "1.1",
            "creator": "MotecLogGenerator",
            "xmlns": "http://www.topografix.com/GPX/1/1"
        })

        trk = ET.SubElement(root, "trk")
        ET.SubElement(trk, "name").text = self.venue_name or "Track Session"
        trkseg = ET.SubElement(trk, "trkseg")

        # Downsample for GPX if high frequency (e.g. max 5Hz for GPX export)
        step = max(1, int(round(lat_chan.avg_frequency() / 5.0))) if lat_chan.avg_frequency() > 5.0 else 1

        for i in range(0, n, step):
            lat = lat_chan.messages[i].value
            lon = lon_chan.messages[i].value

            # Skip zero / invalid lat lon
            if abs(lat) < 0.001 and abs(lon) < 0.001:
                continue

            pt = ET.SubElement(trkseg, "trkpt", {
                "lat": f"{lat:.7f}",
                "lon": f"{lon:.7f}"
            })

            if alt_chan and i < len(alt_chan.messages):
                ET.SubElement(pt, "ele").text = f"{alt_chan.messages[i].value:.2f}"
            if spd_chan and i < len(spd_chan.messages):
                # Speed in m/s for GPX
                spd_ms = spd_chan.messages[i].value / 3.6 if spd_chan.units == "km/h" else spd_chan.messages[i].value
                ET.SubElement(pt, "speed").text = f"{spd_ms:.2f}"

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" ")
        with open(gpx_filename, "w", encoding="utf-8") as f:
            f.write(xml_str)
        return True

    def write_kml(self, kml_filename, data_log):
        """ Export GPS track points as a KML LineString file for Google Earth. """
        lat_chan, lon_chan = self._find_gps_channels(data_log)
        if not lat_chan or not lon_chan:
            return False
        alt_chan = data_log.channels.get("GPS Altitude")

        n = min(len(lat_chan.messages), len(lon_chan.messages))
        if n == 0:
            return False

        coords_str_list = []
        step = max(1, int(round(lat_chan.avg_frequency() / 5.0))) if lat_chan.avg_frequency() > 5.0 else 1

        for i in range(0, n, step):
            lat = lat_chan.messages[i].value
            lon = lon_chan.messages[i].value
            alt = alt_chan.messages[i].value if alt_chan and i < len(alt_chan.messages) else 0.0

            if abs(lat) < 0.001 and abs(lon) < 0.001:
                continue

            coords_str_list.append(f"{lon:.7f},{lat:.7f},{alt:.2f}")

        if not coords_str_list:
            return False

        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        root = ET.Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
        doc = ET.SubElement(root, "Document")
        ET.SubElement(doc, "name").text = self.venue_name or "GPS Track Overlay"

        pm = ET.SubElement(doc, "Placemark")
        ET.SubElement(pm, "name").text = "Track Path"

        style = ET.SubElement(pm, "Style")
        lstyle = ET.SubElement(style, "LineStyle")
        ET.SubElement(lstyle, "color").text = "ff0000ff"  # Red line
        ET.SubElement(lstyle, "width").text = "4"

        ls = ET.SubElement(pm, "LineString")
        ET.SubElement(ls, "extrude").text = "1"
        ET.SubElement(ls, "tessellate").text = "1"
        ET.SubElement(ls, "coordinates").text = "\n".join(coords_str_list)

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" ")
        with open(kml_filename, "w", encoding="utf-8") as f:
            f.write(xml_str)
        return True

