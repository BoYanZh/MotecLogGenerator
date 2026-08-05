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
        dtype = np.float32 if log_channel.data_type is float else np.int32
        freq = int(round(log_channel.avg_frequency()))

        ld_channel = ldChan(
            None,
            meta_ptr=0,
            prev_meta_ptr=0,
            next_meta_ptr=0,
            data_ptr=0,
            data_len=len(log_channel.messages),
            dtype=dtype,
            freq=freq,
            shift=0,
            mul=1,
            scale=1,
            dec=0,
            name=log_channel.name,
            short_name="",
            unit=log_channel.units,
        )
        ld_channel._data = np.array([dtype(msg.value) for msg in log_channel.messages], dtype=dtype)
        self.ld_channels.append(ld_channel)

    def add_all_channels(self, data_log):
        for channel in data_log.channels.values():
            self.add_channel(channel)

    def _finalize_pointers(self):
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
        if not self.ld_channels:
            with open(filename, "wb") as f:
                self.ld_header.write(f, 0)
            return

        self._finalize_pointers()
        ld_data = ldData(self.ld_header, self.ld_channels)
        ld_data.channs[-1].next_meta_ptr = 0
        ld_data.write(filename)

    def write_ldx(self, ldx_filename, laps_info=None):
        import xml.etree.ElementTree as ET

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
        fastest_time_sec = laps_info.get("fastest_time", 0.0) if laps_info else 0.0

        if fastest_time_sec > 0:
            m, s = divmod(fastest_time_sec, 60)
            fastest_time_str = f"{int(m)}:{s:06.3f}"
        else:
            fastest_time_str = "0:00.000"

        add_detail_str("Total Laps", str(total_laps))
        add_detail_str("Fastest Time", fastest_time_str)

        ET.indent(root, space="  ")
        xml_str = ET.tostring(root, encoding="unicode")
        with open(ldx_filename, "w", encoding="utf-8") as f:
            f.write(xml_str)

    @staticmethod
    def _find_gps_channels(data_log):
        lat = data_log.channels.get("GPS Latitude") or data_log.channels.get("Real GPS Latitude")
        lon = data_log.channels.get("GPS Longitude") or data_log.channels.get("Real GPS Longitude")
        return lat, lon

    @staticmethod
    def _downsample_step(channel, target_hz=5.0):
        freq = channel.avg_frequency()
        if freq <= target_hz:
            return 1
        return max(1, int(round(freq / target_hz)))

    @staticmethod
    def _iter_gps_trackpoints(data_log, target_hz=5.0):
        lat_chan, lon_chan = MotecLog._find_gps_channels(data_log)
        if not lat_chan or not lon_chan:
            return
        n = min(len(lat_chan.messages), len(lon_chan.messages))
        if n == 0:
            return
        alt_chan = data_log.channels.get("GPS Altitude")
        spd_chan = data_log.channels.get("Ground Speed")
        step = MotecLog._downsample_step(lat_chan, target_hz)
        for i in range(0, n, step):
            lat = lat_chan.messages[i].value
            lon = lon_chan.messages[i].value
            if abs(lat) < 0.001 and abs(lon) < 0.001:
                continue
            alt = alt_chan.messages[i].value if alt_chan and i < len(alt_chan.messages) else 0.0
            spd = spd_chan.messages[i].value if spd_chan and i < len(spd_chan.messages) else None
            yield lat, lon, alt, spd

    def write_gpx(self, gpx_filename, data_log):
        import xml.etree.ElementTree as ET

        points = list(self._iter_gps_trackpoints(data_log))
        if not points:
            return False

        root = ET.Element("gpx", {
            "version": "1.1",
            "creator": "MotecLogGenerator",
            "xmlns": "http://www.topografix.com/GPX/1/1"
        })
        trk = ET.SubElement(root, "trk")
        ET.SubElement(trk, "name").text = self.venue_name or "Track Session"
        trkseg = ET.SubElement(trk, "trkseg")

        for lat, lon, alt, spd in points:
            pt = ET.SubElement(trkseg, "trkpt", {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}"})
            ET.SubElement(pt, "ele").text = f"{alt:.2f}"
            if spd is not None:
                spd_ms = spd / 3.6
                ET.SubElement(pt, "speed").text = f"{spd_ms:.2f}"

        ET.indent(root, space="  ")
        xml_str = ET.tostring(root, encoding="unicode")
        with open(gpx_filename, "w", encoding="utf-8") as f:
            f.write(xml_str)
        return True

    def write_kml(self, kml_filename, data_log):
        import xml.etree.ElementTree as ET

        points = list(self._iter_gps_trackpoints(data_log))
        if not points:
            return False

        coords_str_list = [f"{lon:.7f},{lat:.7f},{alt:.2f}" for lat, lon, alt, _spd in points]

        root = ET.Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
        doc = ET.SubElement(root, "Document")
        ET.SubElement(doc, "name").text = self.venue_name or "GPS Track Overlay"
        pm = ET.SubElement(doc, "Placemark")
        ET.SubElement(pm, "name").text = "Track Path"
        style = ET.SubElement(pm, "Style")
        lstyle = ET.SubElement(style, "LineStyle")
        ET.SubElement(lstyle, "color").text = "ff0000ff"
        ET.SubElement(lstyle, "width").text = "4"
        ls = ET.SubElement(pm, "LineString")
        ET.SubElement(ls, "extrude").text = "1"
        ET.SubElement(ls, "tessellate").text = "1"
        ET.SubElement(ls, "coordinates").text = "\n".join(coords_str_list)

        ET.indent(root, space="  ")
        xml_str = ET.tostring(root, encoding="unicode")
        with open(kml_filename, "w", encoding="utf-8") as f:
            f.write(xml_str)
        return True

