"""Parser for AIM XRK / XRZ telemetry log files.

Uses the `libxrk` library (https://github.com/m3rlin45/libxrk) to decode the
binary AIM data logger format and populates a DataLog with standard MoTeC
channel names.  XRZ files (zlib-compressed XRK) are handled automatically by
libxrk.

Requires: pip install libxrk
"""

from __future__ import annotations

import datetime

import numpy as np

from core.models import Message
from constants import CHANNEL_ALIASES


def _unit_to_canonical(unit):
    """Normalize a libxrk unit string to a canonical MoTeC unit."""
    unit_l = (unit or "").strip().lower()
    return {
        "km/h": "km/h", "kph": "km/h", "kmh": "km/h", "m/s": "m/s",
        "deg": "deg", "°": "deg", "rpm": "rpm", "g": "G", "m": "m",
        "c": "C", "°c": "C", "bar": "bar", "v": "V", "%": "%",
    }.get(unit_l, unit or "")


def parse_xrk_log(data_log, xrk_file_path, target_lap=None):
    """ Creates channels populated with messages from an AIM .xrk/.xrz log file. """
    try:
        from libxrk import aim_xrk
        from libxrk.base import ChannelMetadata
    except ImportError:
        print("ERROR: 'libxrk' package is required for XRK/XRZ log processing.")
        print("  Install with: pip install libxrk")
        return

    data_log.clear()
    data_log.laps_info = {}

    log = aim_xrk(xrk_file_path)

    # ---- metadata ----
    meta = getattr(log, "metadata", {}) or {}
    venue = str(meta.get("venue", meta.get("Venue", meta.get("track", "")))).strip()
    if venue:
        data_log.metadata["venue_name"] = venue
    driver = str(meta.get("driver", meta.get("Driver", meta.get("racer", "")))).strip()
    if driver:
        data_log.metadata["driver"] = driver
    vehicle = str(meta.get("vehicle", meta.get("Vehicle", meta.get("car", "")))).strip()
    if vehicle:
        data_log.metadata["vehicle_id"] = vehicle
    comment = str(meta.get("comment", meta.get("Comment", ""))).strip()
    if comment:
        data_log.metadata["short_comment"] = comment

    # datetime (AIM stores Unix ms timestamps like RaceChrono)
    for key in ("log_datetime", "datetime", "Log Date/Time", "date_time"):
        raw = meta.get(key)
        if raw:
            try:
                if isinstance(raw, (int, float)) and raw > 1e8:
                    data_log.datetime = datetime.datetime.fromtimestamp(float(raw) / 1000.0)
                else:
                    data_log.datetime = datetime.datetime.strptime(str(raw)[:19], "%Y-%m-%d %H:%M:%S")
                break
            except (ValueError, TypeError, OSError):
                pass

    # ---- channels ----
    for ch_name, table in (log.channels or {}).items():
        if ch_name is None or table is None or table.num_rows == 0:
            continue
        if ch_name.lower() == "timecodes":
            continue

        alias = CHANNEL_ALIASES.get(ch_name)
        if alias is None:
            # try case-insensitive / partial match on common AIM names
            alias = CHANNEL_ALIASES.get(ch_name.strip())
        if alias is None:
            continue

        out_name, unit, dec = alias
        try:
            meta_obj = ChannelMetadata.from_channel_table(table)
        except Exception:
            meta_obj = None

        timecodes_ms = table.column("timecodes").to_numpy()
        values = table.column(ch_name).to_numpy(zero_copy_only=False)

        if out_name not in data_log.channels:
            dec_use = dec if meta_obj is None or not meta_obj.dec_pts else meta_obj.dec_pts
            data_log.add_channel(out_name, unit, float, dec_use)

        # timecodes are milliseconds since log start in AIM logs; anchor at 0
        t0 = float(timecodes_ms[0]) / 1000.0
        msgs = [Message(float(tc) / 1000.0 - t0, float(v)) for tc, v in zip(timecodes_ms, values)]
        data_log.channels[out_name].messages = msgs

    # ---- laps ----
    laps_info = {"laps": [], "total_laps": 0, "fastest_time": 0.0}
    try:
        laps = getattr(log, "laps", None)
        if laps is not None and laps.num_rows > 0:
            lap_nums = laps.column("num").to_pylist()
            starts_ms = laps.column("start_time").to_pylist()
            ends_ms = laps.column("end_time").to_pylist()
            t0_lap = float(starts_ms[0]) / 1000.0 if starts_ms else 0.0
            beacons = []
            for n, s, e in zip(lap_nums, starts_ms, ends_ms):
                start_s = float(s) / 1000.0 - t0_lap
                end_s = float(e) / 1000.0 - t0_lap
                dur = end_s - start_s
                if dur > 0:
                    laps_info["laps"].append({
                        "lap_num": int(n), "start_time": start_s,
                        "end_time": end_s, "duration": dur, "type": "Timed",
                    })
                    beacons.append((start_s, "Start/Finish"))
                    beacons.append((end_s, "Start/Finish"))
                    if laps_info["fastest_time"] == 0.0 or dur < laps_info["fastest_time"]:
                        laps_info["fastest_time"] = dur
            laps_info["total_laps"] = len(laps_info["laps"])
            laps_info["beacons"] = sorted(set(beacons))
    except Exception as e:
        print(f"WARNING: Failed to parse XRK laps: {e}")

    # lap filtering: keep channels within the requested lap window
    if target_lap is not None and str(target_lap).lower() != "all":
        try:
            lap_no = int(target_lap)
            lap_recs = [l for l in laps_info["laps"] if l["lap_num"] == lap_no]
            if lap_recs:
                t_lo = lap_recs[0]["start_time"]
                t_hi = lap_recs[0]["end_time"]
                for name in list(data_log.channels):
                    ch = data_log.channels[name]
                    ch.messages = [m for m in ch.messages if t_lo <= m.timestamp <= t_hi]
        except (ValueError, IndexError):
            print(f"WARNING: Lap '{target_lap}' not found in XRK file; keeping all laps")

    data_log.laps_info = laps_info
