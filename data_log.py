from __future__ import annotations

import datetime
import math

import numpy as np


def _interp_zoh(times_target, times_src, values_src):
    idx = np.searchsorted(times_src, times_target, side="right") - 1
    idx = np.clip(idx, 0, len(values_src) - 1)
    return values_src[idx]


from constants import (
    CH_GROUND_SPEED, CH_CG_ACCEL_LAT, CH_CG_ACCEL_LON, CH_GPS_LATITUDE, CH_GPS_LONGITUDE,
    CH_GPS_HEADING, CH_GPS_ALTITUDE, CH_GPS_SATS, CH_LAP_NUMBER, CH_THROTTLE_POS,
    CH_BRAKE_PRESS, CH_BRAKE_POS, CH_ENGINE_RPM, CH_STEERING_ANGLE, CH_COOLANT_TEMP,
    CH_ENGINE_OIL_TEMP, CH_GEAR, CH_YAW_RATE, CH_SLIP_ANGLE_FL, CH_SLIP_ANGLE_FR,
    CH_SLIP_ANGLE_RL, CH_SLIP_ANGLE_RR, CH_UNDERSTEER_INDEX, CH_G_COMBINED,
    CHANNEL_ALIASES
)

DISCRETE_CHANNELS = {CH_GEAR, CH_LAP_NUMBER, "GPS Fix", CH_GPS_SATS}


def _parse_vbo_latlon(val_str):
    sign = -1.0 if val_str.startswith("-") else 1.0
    val_str = val_str.lstrip("+-")
    dot_idx = val_str.find(".")
    if dot_idx < 0:
        return 0.0
    deg_str = val_str[:dot_idx - 2]
    min_str = val_str[dot_idx - 2:]
    deg = float(deg_str) if deg_str else 0.0
    minutes = float(min_str)
    return sign * (deg + minutes / 60.0)


def _parse_vbo_time(val_str):
    try:
        t_float = float(val_str)
        hh = int(t_float // 10000)
        mm = int((t_float % 10000) // 100)
        ss = t_float % 100
        return hh * 3600.0 + mm * 60.0 + ss
    except ValueError:
        return 0.0


class DataLog(object):
    channels: dict[str, Channel]
    """ Container for storing log data which contains a set of channels with time series data."""

    def __init__(self, name=""):
        self.name = name
        self.channels = {}
        self.datetime = None
        self.metadata = {}
        self.traps = []

    def clear(self):
        self.channels = {}
        self.datetime = None
        self.metadata = {}
        self.traps = []

    def add_channel(self, name, units, data_type, decimals, initial_message=None):
        if any(k in name.lower() for k in ["latitude", "longitude", "lat", "lon"]):
            decimals = max(decimals, 7)
        msg = [] if not initial_message else [initial_message]
        self.channels[name] = Channel(name, units, data_type, decimals, msg)

    def start(self):
        """ Returns the earliest timestamp from all existing channels [s]. """
        t = math.inf
        for name, channel in self.channels.items():
            t = min(t, channel.start())

        return t if t != math.inf else 0.0

    def end(self):
        """ Returns the latest timestamp from all existing channels [s]. """
        end = -math.inf
        for name, channel in self.channels.items():
            end = max(end, channel.end())

        return end if end != -math.inf else 0.0

    def duration(self):
        """ Returns the duration of the log [s]. """
        s = self.start()
        e = self.end()
        return max(0.0, e - s)

    def detect_native_frequency(self) -> float:
        """ Detects the primary native sampling frequency of the log data. """
        # 1. Check device_update_rate channel if present
        if "device_update_rate" in self.channels:
            vals = [m.value for m in self.channels["device_update_rate"].messages if m.value > 0]
            if vals:
                rate = float(np.mean(vals))
                for std_rate in [10.0, 20.0, 25.0, 50.0, 100.0]:
                    if abs(rate - std_rate) / std_rate <= 0.10:
                        return std_rate

        # 2. Inspect key physical channels
        candidates = ["Ground Speed", "GPS Latitude", "Running Time", "CG Accel Lateral", "Engine RPM"]
        raw_freq = 0.0
        for name in candidates:
            if name in self.channels and len(self.channels[name].messages) > 1:
                raw_freq = self.channels[name].avg_frequency()
                if raw_freq > 0:
                    break
        if raw_freq <= 0:
            all_freqs = [c.avg_frequency() for c in self.channels.values() if len(c.messages) > 1]
            raw_freq = max(all_freqs) if all_freqs else 25.0

        # Snap to standard logging rates (up to 50Hz for auto-detection)
        for std_rate in [10.0, 20.0, 25.0, 50.0]:
            if abs(raw_freq - std_rate) / std_rate <= 0.10:
                return std_rate

        if raw_freq > 50.0:
            return 50.0

        return max(1.0, round(raw_freq, 1))

    def resample(self, frequency="auto"):
        """ Resamples all channels such that all messages occur at a fixed frequency.

        frequency: float, int, or 'auto' (detects native sample rate, e.g. 20Hz, 25Hz, 50Hz).
        Returns the frequency used for resampling.
        """
        if isinstance(frequency, str) and frequency.lower() == "auto":
            frequency = self.detect_native_frequency()
        elif frequency is None or (isinstance(frequency, (int, float)) and frequency <= 0):
            frequency = self.detect_native_frequency()
        else:
            frequency = float(frequency)

        start = self.start()
        end = self.end()
        for channel_name in self.channels:
            self.channels[channel_name].resample(start, end, frequency)
        return frequency

    def detect_beacons(self, min_speed_kmh=30.0, min_time_sec=15.0):
        """ Detects trap / sector crossing timestamps from GPS coordinates and traps metadata. """
        if not getattr(self, "traps", None):
            return []
        if "GPS Latitude" not in self.channels or "GPS Longitude" not in self.channels:
            return []

        lat_m = self.channels["GPS Latitude"].messages
        lon_m = self.channels["GPS Longitude"].messages
        if not lat_m or not lon_m or len(lat_m) != len(lon_m):
            return []

        spd_m = self.channels.get("Ground Speed")
        spd_vals = np.array([m.value for m in spd_m.messages]) if spd_m else None

        lat = np.array([m.value for m in lat_m])
        lon = np.array([m.value for m in lon_m])
        times = np.array([m.timestamp for m in lat_m])
        dur = self.duration()

        beacons = []
        for t in self.traps:
            t_lat = t["lat"]
            t_lon = t["lon"]
            t_name = t["name"]

            # Flat earth distance approximation (meters around ~35-45 deg lat)
            d_lat = (lat - t_lat) * 111000.0
            d_lon = (lon - t_lon) * 86000.0
            dist = np.sqrt(d_lat**2 + d_lon**2)

            for i in range(1, len(dist) - 1):
                # Speed & boundary guards: ignore false crossings while stationary/pitting
                # or near session start/end
                if spd_vals is not None and spd_vals[i] < min_speed_kmh:
                    continue
                if times[i] < min_time_sec or (dur > 0 and (dur - times[i]) < 5.0):
                    continue

                if dist[i] < 30.0 and dist[i] <= dist[i - 1] and dist[i] <= dist[i + 1]:
                    beacons.append((float(times[i]), t_name))

        beacons.sort(key=lambda x: x[0])
        return beacons

    def calculate_math_channels(self, g_source="auto"):
        """
        g_source: "auto" (default), "sensor", or "calc"
          - "auto": Use raw IMU sensor G channels if present; otherwise derive from GPS.
          - "sensor": Only use raw IMU sensor G channels. Do not derive G from GPS.
          - "calc": Force deriving G channels from GPS Speed + Yaw Rate, overriding sensor channels.
        """
        self._derive_yaw_rate_from_gps_heading()

        if g_source == "calc":
            if "CG Accel Lateral" in self.channels:
                del self.channels["CG Accel Lateral"]
            if "CG Accel Longitudinal" in self.channels:
                del self.channels["CG Accel Longitudinal"]
            self._derive_cg_accel_lateral(force=True)
            self._derive_cg_accel_longitudinal(force=True)
        elif g_source == "sensor":
            pass
        else:  # "auto"
            self._derive_cg_accel_lateral(force=False)
            self._derive_cg_accel_longitudinal(force=False)

        self._derive_smoothed_accel()
        self._calculate_kinematics()
        self._calculate_g_sum()
        self._derive_brake_pos()
        self._calculate_input_rates()
        self._mirror_throttle_accel()

    def _derive_smoothed_accel(self, window_sec=0.5):
        """ Derive 0.5s moving average smoothed G channels for clean G-G diagrams in MoTeC. """
        for raw_name, smooth_name in [("CG Accel Lateral", "CG Accel Lateral Smooth"),
                                      ("CG Accel Longitudinal", "CG Accel Long Smooth")]:
            if raw_name in self.channels and smooth_name not in self.channels:
                ch = self.channels[raw_name]
                if len(ch.messages) < 5:
                    continue
                freq = ch.avg_frequency()
                w = max(1, int(window_sec * freq))
                vals = np.array([m.value for m in ch.messages], dtype=np.float64)

                padded = np.pad(vals, (w // 2, w - 1 - w // 2), mode="edge")
                windows = np.lib.stride_tricks.sliding_window_view(padded, w)
                smoothed = np.mean(windows, axis=1)[:len(vals)]

                self.add_channel(smooth_name, ch.units, float, ch.decimals)
                self.channels[smooth_name].messages = [
                    Message(ch.messages[i].timestamp, float(smoothed[i])) for i in range(len(vals))
                ]

    def _derive_cg_accel_lateral(self, force=False):
        if not force and "CG Accel Lateral" in self.channels:
            return
        if "Ground Speed" not in self.channels or "Chassis Yaw Rate" not in self.channels:
            return
        spd_chan = self.channels["Ground Speed"]
        yaw_chan = self.channels["Chassis Yaw Rate"]
        n = len(spd_chan.messages)
        if n < 1:
            return
        t_arr = np.array([m.timestamp for m in spd_chan.messages])
        vx_ms = np.array([m.value / 3.6 if spd_chan.units == "km/h" else m.value for m in spd_chan.messages])
        yr_rad = np.radians([m.value for m in yaw_chan.messages])
        ay = (vx_ms * yr_rad) / 9.80665
        self.add_channel("CG Accel Lateral", "G", float, 2)
        self.channels["CG Accel Lateral"].messages = [Message(t_arr[i], ay[i]) for i in range(n)]

    def _derive_cg_accel_longitudinal(self, force=False):
        if not force and "CG Accel Longitudinal" in self.channels:
            return
        if "Ground Speed" not in self.channels:
            return
        spd_chan = self.channels["Ground Speed"]
        n = len(spd_chan.messages)
        if n < 2:
            return
        t_arr = np.array([m.timestamp for m in spd_chan.messages])
        vx_ms = np.array([m.value / 3.6 if spd_chan.units == "km/h" else m.value for m in spd_chan.messages])
        ax = np.gradient(vx_ms, t_arr) / 9.80665
        self.add_channel("CG Accel Longitudinal", "G", float, 2)
        self.channels["CG Accel Longitudinal"].messages = [Message(t_arr[i], ax[i]) for i in range(n)]

    # --- GR86 vehicle dynamics constants ---
    # These are tuned specifically for the Toyota GR86 / Subaru BRZ (2022+).
    # The tire slip angle and understeer index channels will be incorrect for
    # any other vehicle. To adapt, replace these with values for your car.
    _GR86_STEERING_RATIO = 13.5          # steering wheel angle : road wheel angle
    _GR86_WHEELBASE_M = 2.575            # distance between front and rear axles
    _GR86_CG_TO_FRONT_AXLE_M = 1.25      # distance from center of gravity to front axle
    _GR86_CG_TO_REAR_AXLE_M = 1.325      # distance from center of gravity to rear axle
    _GR86_LAT_VEL_TAU_S = 2.0            # time constant for lateral velocity complementary filter

    def _calculate_kinematics(self):
        required = ["Ground Speed", "CG Accel Lateral", "Chassis Yaw Rate"]
        if not all(r in self.channels for r in required):
            return
        vx_chan = self.channels["Ground Speed"]
        n = min(len(self.channels[r].messages) for r in required)
        if n < 2:
            return
        time = np.array([m.timestamp for m in vx_chan.messages[:n]])
        vx = np.array([m.value for m in vx_chan.messages[:n]])
        if vx_chan.units == "km/h":
            vx /= 3.6
        ay = np.array([m.value * 9.80665 for m in self.channels["CG Accel Lateral"].messages[:n]])
        yaw_rate_degs = np.array([m.value for m in self.channels["Chassis Yaw Rate"].messages[:n]])
        yaw_rate = np.radians(yaw_rate_degs * -1.0)

        dt = np.zeros(n)
        dt[1:] = np.diff(time)

        vy = np.zeros(n)
        beta = np.zeros(n)
        tau = self._GR86_LAT_VEL_TAU_S

        for i in range(1, n):
            vy_dot = ay[i] - (vx[i] * yaw_rate[i])
            alpha = np.exp(-dt[i] / tau)
            vy[i] = (vy[i - 1] + vy_dot * dt[i]) * alpha
            if abs(ay[i]) < 0.49 and abs(yaw_rate_degs[i]) < 1.0:
                vy[i] = 0.0
            if vx[i] > 5.0:
                beta[i] = np.arctan2(vy[i], vx[i])

        ratio = self._GR86_STEERING_RATIO
        wheelbase = self._GR86_WHEELBASE_M
        lf = self._GR86_CG_TO_FRONT_AXLE_M
        lr = self._GR86_CG_TO_REAR_AXLE_M

        slip_f = np.zeros(n)
        slip_r = np.zeros(n)
        steer_rad = np.zeros(n)
        if "Steering Angle" in self.channels:
            steer_deg = np.array([m.value for m in self.channels["Steering Angle"].messages])
            steer_rad = np.radians(steer_deg / ratio)

        for i in range(n):
            if vx[i] > 5.0:
                slip_f[i] = np.degrees(steer_rad[i] - np.arctan2(vy[i] + yaw_rate[i] * lf, vx[i]))
                slip_r[i] = np.degrees(-np.arctan2(vy[i] - yaw_rate[i] * lr, vx[i]))

        for name in ("Tire Slip Angle FL", "Tire Slip Angle FR", "Tire Slip Angle RL", "Tire Slip Angle RR"):
            self.add_channel(name, "deg", float, 2)
            src_data = slip_f if "F" in name else slip_r
            self.channels[name].messages = [Message(time[i], src_data[i]) for i in range(n)]

        us_index = np.zeros(n)
        for i in range(n):
            if vx[i] > 5.0:
                us_index[i] = np.degrees(steer_rad[i]) - np.degrees(wheelbase * yaw_rate[i] / vx[i])
        self.add_channel("Understeer Index", "deg", float, 2)
        self.channels["Understeer Index"].messages = [Message(time[i], us_index[i]) for i in range(n)]

    def _calculate_g_sum(self):
        if "CG Accel Longitudinal" not in self.channels or "CG Accel Lateral" not in self.channels:
            return
        ax_msgs = self.channels["CG Accel Longitudinal"].messages
        ay_msgs = self.channels["CG Accel Lateral"].messages
        n = min(len(ax_msgs), len(ay_msgs))
        if n < 2:
            return
        time_g = [ax_msgs[i].timestamp for i in range(n)]
        ax = np.array([ax_msgs[i].value for i in range(n)])
        ay_g = np.array([ay_msgs[i].value for i in range(n)])
        g_sum = np.sqrt(ax ** 2 + ay_g ** 2)
        self.add_channel("G Force Combined", "G", float, 2)
        self.channels["G Force Combined"].messages = [Message(time_g[i], g_sum[i]) for i in range(n)]

    def _derive_brake_pos(self):
        # GR86 empirical: brake master cylinder pressure ~96 kPa per 1% brake pedal position
        if "Brake Press" not in self.channels or "Brake Pos" in self.channels:
            return
        press_chan = self.channels["Brake Press"]
        n = len(press_chan.messages)
        if n < 1:
            return
        time_p = [m.timestamp for m in press_chan.messages]
        press_vals = np.array([m.value for m in press_chan.messages])
        if press_chan.units == "bar":
            press_vals *= 100.0
        bpos = np.clip(press_vals / 96.0, 0.0, 100.0)
        self.add_channel("Brake Pos", "%", float, 2)
        self.channels["Brake Pos"].messages = [Message(time_p[i], bpos[i]) for i in range(n)]

    def _calculate_input_rates(self):
        self.__calculate_rate("Steering Angle", "deg/s")
        self.__calculate_rate("Throttle Pos", "%/s")
        self.__calculate_rate("Brake Pos", "%/s")

    def _mirror_throttle_accel(self):
        if "Throttle Pos" not in self.channels and "Accelerator Pos" in self.channels:
            src = self.channels["Accelerator Pos"]
            self.add_channel("Throttle Pos", "%", float, 2)
            self.channels["Throttle Pos"].messages = [Message(m.timestamp, m.value) for m in src.messages]

    def __calculate_rate(self, channel_name, unit):
        """ Internal helper to calculate the rate of change for a channel. """
        if channel_name in self.channels:
            chan = self.channels[channel_name]
            vals = np.array([m.value for m in chan.messages])
            times = np.array([m.timestamp for m in chan.messages])
            if len(times) < 2:
                return
            rate = np.zeros(len(times))
            dt = np.diff(times)
            dt[dt == 0] = 0.001 # Prevent div by zero
            rate[1:] = np.diff(vals) / dt

            new_name = channel_name + " Rate"
            self.add_channel(new_name, unit, float, 2)
            self.channels[new_name].messages = [Message(times[i], rate[i]) for i in range(len(times))]

    def _derive_yaw_rate_from_gps_heading(self):
        if "Chassis Yaw Rate" in self.channels:
            return
        gps_h_chan = self.channels.get("GPS Heading")
        if not gps_h_chan or len(gps_h_chan.messages) < 2:
            return
        times = np.array([m.timestamp for m in gps_h_chan.messages])
        headings = np.array([m.value for m in gps_h_chan.messages])
        h_unwrapped = np.unwrap(np.radians(headings))
        h_deg = np.degrees(h_unwrapped)
        yaw_rate_val = -np.gradient(h_deg, times)
        freq = gps_h_chan.avg_frequency()
        w = max(1, int(0.2 * freq))
        if len(yaw_rate_val) >= w and w > 1:
            padded = np.pad(yaw_rate_val, (w // 2, w - 1 - w // 2), mode="edge")
            yaw_rate_val = np.mean(np.lib.stride_tricks.sliding_window_view(padded, w), axis=1)[:len(yaw_rate_val)]
        spd_chan = self.channels.get("Ground Speed")
        if spd_chan and len(spd_chan.messages) == len(times):
            v_vals = np.array([m.value for m in spd_chan.messages])
            if spd_chan.units == "mph":
                v_vals *= 1.60934
            yaw_rate_val[v_vals < 5.0] = 0.0
        yaw_rate_val = np.clip(yaw_rate_val, -150.0, 150.0)
        self.add_channel("Chassis Yaw Rate", "deg/s", float, 2)
        self.channels["Chassis Yaw Rate"].messages = [Message(times[i], yaw_rate_val[i]) for i in range(len(times))]

    def _extract_datetime_from_text(self, log_lines, file_path=""):
        import datetime
        import os
        import re

        # 1. Filename pattern: session_YYYYMMDD_HHMMSS or YYYYMMDD_HHMM
        if file_path:
            filename = os.path.basename(file_path)
            match = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})?", filename)
            if match:
                groups = match.groups()
                y, m, d, hh, mm = map(int, groups[:5])
                ss = int(groups[5]) if groups[5] else 0
                try:
                    return datetime.datetime(y, m, d, hh, mm, ss)
                except ValueError:
                    pass

        if not log_lines:
            return None

        date_str = ""
        time_str = ""

        import csv

        # 2. Check header and data lines
        for line in log_lines[:50]:
            line_clean = line.strip()
            if not line_clean:
                continue

            # Check candump format: (1630268615.800257)
            if line_clean.startswith("(") and ")" in line_clean:
                try:
                    stamp = float(line_clean.split(")")[0][1:])
                    if stamp > 1e8:
                        return datetime.datetime.fromtimestamp(stamp)
                except ValueError:
                    pass

            try:
                parts = [p.strip().strip('"').strip("'") for p in next(csv.reader([line_clean]))]
            except Exception:
                parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]

            if len(parts) >= 2:
                k = parts[0].lower()
                v = parts[1]
                if k in ["date", "log date", "created"]:
                    date_str = v
                elif k in ["time", "log time"] and (":" in v or "am" in v.lower() or "pm" in v.lower()):
                    time_str = v

            if parts and parts[0]:
                try:
                    stamp = float(parts[0])
                    if stamp > 1e8:
                        return datetime.datetime.fromtimestamp(stamp)
                except ValueError:
                    pass

        if date_str:
            dt_raw = f"{date_str} {time_str}".strip()
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%m/%d/%y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
                "%A, %B %d, %Y %I:%M %p",
                "%A, %B %d, %Y %H:%M",
                "%Y-%m-%d",
                "%m/%d/%y",
                "%m/%d/%Y",
                "%d/%m/%Y"
            ]
            for fmt in formats:
                try:
                    return datetime.datetime.strptime(dt_raw, fmt)
                except ValueError:
                    pass

        return None

    def _extract_metadata(self, log_lines=None, file_path=""):
        import os
        metadata = {
            "driver": "",
            "vehicle_id": "",
            "venue_name": "",
            "event_name": "",
            "event_session": "",
            "short_comment": ""
        }

        filename = os.path.basename(file_path) if file_path else ""

        # 1. Filename pattern (_&_ split)
        if filename and "_&_" in filename:
            parts = filename.replace(".csv", "").replace(".rcz", "").split("_&_")
            if len(parts) >= 4:
                metadata["venue_name"] = parts[0]
                metadata["vehicle_id"] = parts[1]
                metadata["driver"] = parts[2]
                metadata["event_session"] = parts[3]

        # 2. Check header lines if available
        if log_lines:
            for line in log_lines[:30]:
                line_clean = line.strip().strip('"').strip("'")
                parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]
                if len(parts) >= 2:
                    k = parts[0].lower()
                    v = parts[1]
                    if not v:
                        continue
                    if k in ["driver", "racer"]:
                        metadata["driver"] = v
                    elif k in ["vehicle", "car"]:
                        metadata["vehicle_id"] = v
                    elif k in ["session"]:
                        metadata["event_session"] = v
                    elif k in ["event", "championship"]:
                        metadata["event_name"] = v
                    elif k in ["venue", "track"]:
                        metadata["venue_name"] = v
                    elif k in ["comment", "notes"]:
                        metadata["short_comment"] = v

        # 3. Fallback venue inference from filename
        if not metadata["venue_name"] and filename:
            fn_lower = filename.lower()
            if "thunder" in fn_lower:
                metadata["venue_name"] = "Thunderhill Raceway Park"
            elif "laguna" in fn_lower:
                metadata["venue_name"] = "Laguna Seca"
            elif "sonoma" in fn_lower:
                metadata["venue_name"] = "Sonoma Raceway"

        self.metadata.update({k: v for k, v in metadata.items() if v})

    def from_can_log(self, log_lines, can_db):
        """ Creates channels populated with messages from a candump file and can database.

        This will create a channel for each entry in the database that has messages present in the
        log.

        log_lines: List, containing candump log lines (recorded with 'candump' with '-l')
        can_db: cantools.database
        """
        self.clear()
        self.datetime = self._extract_datetime_from_text(log_lines)

        # Cache all the frame ids in the database for quick lookups
        known_ids = set()
        for msg in can_db.messages:
            known_ids.add(msg.frame_id)

        for line in log_lines:
            stamp, bus, id, data = self.__parse_can_log_line(line)
            if stamp is None:
                continue

            if id not in known_ids:
                continue

            db_msg = can_db.get_message_by_frame_id(id)
            msg_decoded = can_db.decode_message(id, data)

            for msg, signal in zip(msg_decoded.items(), db_msg.signals):
                name = msg[0]
                value = msg[1]

                if name in self.channels:
                    self.channels[name].messages.append(Message(stamp, value))
                else:
                    self.add_channel(name, signal.unit, float, 3, Message(stamp, value))

    def from_csv_log(self, log_lines):
        """ Creates channels populated with messages from a CSV log file.

        This will create a channel for each column in the CSV file, with the name of that channel
        taken from the CSV header. All channels will be created without any units. Any non numeric data
        will be ignored, and that channel will be removed. The first column of data must be time

        log_lines: List, containing CSV log lines
        """
        self.clear()

        # 1. Dynamically locate Data Header Line
        data_header_idx = 0
        import csv
        for i, line in enumerate(log_lines):
            line_clean = line.strip().strip('"').strip("'")
            if not line_clean:
                continue
            try:
                parts = [p.strip().strip('"').strip("'") for p in next(csv.reader([line_clean]))]
            except Exception:
                parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]
            if len(parts) >= 2 and parts[0].lower() in ("time", "time (s)", "timestamp"):
                data_header_idx = i
                break

        # Get the channel names, ignore the first column as it is assumed to be time
        header = log_lines[data_header_idx].strip("\n")
        channel_names = [name.strip().strip('"').strip("'") for name in header.split(",")[1:]]

        # We'll keep a map of names and column numbers for easy channel lookups when parsing rows
        i = 0
        channel_dict = {}
        for name in channel_names:
            self.add_channel(name, "", float, 0)

            channel_dict[name] = i
            i += 1

        # Go through each line grabbing all the channel values
        for line in log_lines[data_header_idx + 1:]:
            line = line.strip("\n")
            values = line.split(",")

            # Timestamp is the first element
            t = float(values[0])

            # Grab each remaining channel value. We keep a map of all the channel names and column
            # numbers we are retrieving, so we will look at that to determine which columns to read.
            # If we fail to read an entry in any column, we will delete that channel entirely.
            invalid_channels = []
            for name, i in channel_dict.items():
                # We'll only parse numeric data
                try:
                    val = float(values[i + 1])
                    message = Message(t, val)
                    self.channels[name].messages.append(message)

                    val_text_split = values[i + 1].split(".")
                    decimals_present = 0 if len(val_text_split) == 1 else len(val_text_split[1])
                    self.channels[name].decimals = max(decimals_present, self.channels[name].decimals)
                except ValueError:
                    print("WARNING: Found non numeric values for channel %s, removing channel" % \
                        name)
                    invalid_channels.append(name)

            for name in invalid_channels:
                del channel_dict[name]
                del self.channels[name]
            if invalid_channels:
                channel_dict = {name: idx for idx, name in enumerate(channel_dict)}

    def from_racechrono_log(self, log_lines, target_lap=None):
        """ Creates channels populated with messages from a RaceChrono CSV log file.

        This maps standard RaceChrono columns to MoTeC-compatible names and units.
        
        log_lines: List, containing CSV log lines
        target_lap: String, int, or None. If None or 'all', processes all laps in the session.
        """
        self.clear()
        self.laps_info = {}
        file_p = getattr(self, "log_file_path", "")
        self.datetime = self._extract_datetime_from_text(log_lines, file_p)
        self._extract_metadata(log_lines, file_p)

        if not log_lines:
            return

        # Find the table header row (typically starts with "Time (s)", "timestamp", or "Time" and has > 2 columns)
        header_idx = -1
        import csv
        for i, line in enumerate(log_lines):
            line_clean = line.strip().strip('"').strip("'")
            if not line_clean:
                continue
            try:
                parts = [p.strip().strip('"').strip("'") for p in next(csv.reader([line_clean]))]
            except Exception:
                parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]
            if parts and len(parts) > 2:
                first_col = parts[0].lower()
                if first_col in ["time (s)", "timestamp", "time"]:
                    header_idx = i
                    break

        if header_idx == -1:
            print("ERROR: Could not find 'Time (s)', 'timestamp', or 'Time' column in CSV log.")
            return

        header = log_lines[header_idx].strip("\n")
        # Split and strip any quotes from header names
        raw_headers = [h.strip().strip('"').strip("'") for h in header.split(",")]

        # Check for unit row (row right after header row if it contains units like "s", "mph", "km/h")
        unit_row = []
        if header_idx + 1 < len(log_lines):
            next_line = log_lines[header_idx + 1].strip("\n")
            splits = [s.strip().strip('"').strip("'") for s in next_line.split(",")]
            if splits and splits[0].lower() in ["s", "sec", "seconds", ""]:
                unit_row = splits

        channel_names = raw_headers[1:]

        # Check if yaw_rate exists, otherwise check for y_rate_of_rotation / z_rate_of_rotation / GPS Gyro
        has_yaw = any(h.lower() in ["yaw_rate", "gps gyro", "chassis yaw rate"] for h in channel_names)

        # Base mapping from typical RaceChrono / AiM / MoTeC columns to MoTeC terminology and units.
        rc_to_motec_map = {
            "lap_number": {"name": "Lap Number", "units": ""},
            "elapsed_time": {"name": "Running Time", "units": "s"},
            "distance_traveled": {"name": "Corr Dist", "units": "m"},
            "accuracy": {"name": "GPS Accuracy", "units": "m"},
            "altitude": {"name": "GPS Altitude", "units": "m"},
            "bearing": {"name": "GPS Heading", "units": "deg"},
            "device_battery_level": {"name": "Device Battery", "units": "%"},
            "fix_type": {"name": "GPS Fix", "units": ""},
            "latitude": {"name": "GPS Latitude", "units": "deg"},
            "longitude": {"name": "GPS Longitude", "units": "deg"},
            "satellites": {"name": "GPS Satellites", "units": ""},
            "speed": {"name": "Ground Speed", "units": "km/h"},
            "combined_acc": {"name": "G Force Combined", "units": "G"},
            "lateral_acc": {"name": "CG Accel Lateral", "units": "G"},
            "lean_angle": {"name": "Lean Angle", "units": "deg"},
            "longitudinal_acc": {"name": "CG Accel Longitudinal", "units": "G"},
            "accelerator_pos": {"name": "Throttle Pos", "units": "%"},
            "brake_pos": {"name": "Brake Pos", "units": "%"},
            "brake_pressure": {"name": "Brake Press", "units": "kPa"},
            "coolant_temp": {"name": "Coolant Temp", "units": "C"},
            "engine_oil_temp": {"name": "Engine Oil Temp", "units": "C"},
            "rpm": {"name": "Engine RPM", "units": "rpm"},
            "steering_angle": {"name": "Steering Angle", "units": "deg"},
            "yaw_rate": {"name": "Chassis Yaw Rate", "units": "deg/s"},
            "gear": {"name": "Gear", "units": ""},
            "gear_position": {"name": "Gear", "units": ""},
            # AiM Solo / RaceStudio CSV Mappings
            "gps speed": {"name": "Ground Speed", "units": "km/h"},
            "gps latacc": {"name": "CG Accel Lateral", "units": "G"},
            "gps lonacc": {"name": "CG Accel Longitudinal", "units": "G"},
            "gps gyro": {"name": "Chassis Yaw Rate", "units": "deg/s"},
            "gps lat": {"name": "GPS Latitude", "units": "deg"},
            "gps lon": {"name": "GPS Longitude", "units": "deg"},
            "gps altitude": {"name": "GPS Altitude", "units": "m"},
            "gps heading": {"name": "GPS Heading", "units": "deg"},
            "gps posaccuracy": {"name": "GPS Accuracy", "units": "m"},
            "pps": {"name": "Throttle Pos", "units": "%"},
            "steerangle": {"name": "Steering Angle", "units": "deg"},
            "brakepress": {"name": "Brake Press", "units": "kPa"},
            "oiltemp": {"name": "Engine Oil Temp", "units": "C"},
            "ect": {"name": "Coolant Temp", "units": "C"},
            "intake air temp": {"name": "Intake Temp", "units": "C"},
            "oilpressure0": {"name": "Engine Oil Press", "units": "kPa"},
            "yawrate": {"name": "Chassis Yaw Rate", "units": "deg/s"},
            "wheelspeedfl": {"name": "Wheel Speed FL", "units": "km/h"},
            "wheelspeedfr": {"name": "Wheel Speed FR", "units": "km/h"},
            "wheelspeedrl": {"name": "Wheel Speed RL", "units": "km/h"},
            "wheelspeedrr": {"name": "Wheel Speed RR", "units": "km/h"},
            "speedv": {"name": "Vehicle Speed", "units": "km/h"},
            # MoTeC CSV Mappings
            "corr speed": {"name": "Ground Speed", "units": "km/h"},
            "gps latitude": {"name": "GPS Latitude", "units": "deg"},
            "gps longitude": {"name": "GPS Longitude", "units": "deg"},
            "corr dist": {"name": "Corr Dist", "units": "m"},
        }

        # Fallback for yaw rate if not named "yaw_rate" or "gps gyro"
        if not has_yaw:
            if "z_rate_of_rotation" in channel_names:
                rc_to_motec_map["z_rate_of_rotation"] = {"name": "Chassis Yaw Rate", "units": "deg/s"}
            elif "y_rate_of_rotation" in channel_names:
                rc_to_motec_map["y_rate_of_rotation"] = {"name": "Chassis Yaw Rate", "units": "deg/s"}

        # Explicitly ignore uncalibrated raw IMU channels
        ignored_columns = {"x_acc", "y_acc", "z_acc"}

        active_columns = []
        lap_number_idx = -1
        col_unit_map = {}

        for i, raw_name in enumerate(channel_names):
            if not raw_name or raw_name in ignored_columns:
                continue

            raw_lower = raw_name.lower().strip().replace("_", " ").replace("-", " ")
            if raw_lower == "lap_number" or raw_lower == "lap":
                lap_number_idx = i + 1

            unit_str = ""
            if unit_row and i + 1 < len(unit_row):
                unit_str = unit_row[i + 1]

            if raw_lower in rc_to_motec_map:
                motec_name = rc_to_motec_map[raw_lower]["name"]
                motec_units = rc_to_motec_map[raw_lower]["units"]
            else:
                # Fallback to general parsing if no exact map is found
                if " (" in raw_name and raw_name.endswith(")"):
                    motec_name, motec_units = raw_name.rsplit(" (", 1)
                    motec_units = motec_units[:-1]
                else:
                    motec_name = raw_name
                    motec_units = unit_str

            col_unit_map[raw_name] = unit_str.lower()

            if motec_name not in self.channels:
                self.add_channel(motec_name, motec_units, float, 0)
                active_columns.append((i, motec_name, raw_name))

        # First pass to parse all valid data rows
        valid_lines = []
        for line in log_lines[header_idx + 1:]:
            line_clean = line.strip("\n")
            if not line_clean:
                continue

            # Skip unit rows (typically right after headers and start with empty timestamp or strings)
            values = line_clean.split(",")
            if not values[0]:
                continue

            try:
                float(values[0].strip().strip('"').strip("'"))
            except ValueError:
                continue

            valid_lines.append(values)

        # Parse target_lap filter if specified
        target_lap_str = None
        if target_lap is not None and str(target_lap).lower() != "all":
            target_lap_str = str(target_lap).strip()

        # Track lap timing metadata
        laps_timing = {}

        for values in valid_lines:
            if lap_number_idx != -1 and lap_number_idx < len(values):
                lap_val = values[lap_number_idx].strip().strip('"').strip("'")
                if target_lap_str and lap_val != target_lap_str:
                    continue
            else:
                lap_val = "1"

            try:
                t = float(values[0].strip().strip('"').strip("'"))
            except ValueError:
                continue

            # Record lap start/end times
            if lap_val not in laps_timing:
                laps_timing[lap_val] = {"start_time": t, "end_time": t, "messages_count": 0}
            laps_timing[lap_val]["end_time"] = t
            laps_timing[lap_val]["messages_count"] += 1

            for col_idx, name, raw_name in active_columns:
                if col_idx + 1 >= len(values):
                    continue

                val_str = values[col_idx + 1].strip().strip('"').strip("'")
                if not val_str:
                    continue

                try:
                    val = float(val_str)

                    # Convert speed based on unit or column name
                    raw_lower = raw_name.lower().strip()
                    unit_lower = col_unit_map.get(raw_name, "")
                    if raw_lower == "speed":
                        val *= 3.6
                    elif unit_lower == "mph":
                        val *= 1.60934
                    elif unit_lower == "psi" and name == "Brake Press":
                        val *= 6.89476
                    elif unit_lower == "f":
                        val = (val - 32.0) * (5.0 / 9.0)
                    elif unit_lower == "bar" and "press" in name.lower():
                        val *= 100.0
                    elif raw_lower in ("yaw_rate", "z_rate_of_rotation", "y_rate_of_rotation") and name == "Chassis Yaw Rate":
                        val *= -1.0

                    message = Message(t, val)
                    self.channels[name].messages.append(message)

                    val_text_split = val_str.split(".")
                    decimals_present = 0 if len(val_text_split) == 1 else len(val_text_split[1])
                    self.channels[name].decimals = max(decimals_present, self.channels[name].decimals)
                except ValueError:
                    pass

        # Fallback: derive Chassis Yaw Rate from GPS Heading derivative if missing
        self._derive_yaw_rate_from_gps_heading()

        # Construct laps_info summary (with Out Lap and In Lap)
        header_beacons = []
        import csv
        for line in log_lines[:header_idx]:
            line_clean = line.strip().strip('"').strip("'")
            try:
                parts = [p.strip().strip('"').strip("'") for p in next(csv.reader([line_clean]))]
            except Exception:
                parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]
            if parts and parts[0].lower() == "beacon markers":
                for val in parts[1:]:
                    try:
                        header_beacons.append(float(val))
                    except ValueError:
                        pass

        if header_beacons:
            lap_items = []
            fastest_lap = None
            fastest_dur = math.inf
            start_t = 0.0
            total_beacons = len(header_beacons)
            for idx, btime in enumerate(header_beacons):
                dur = btime - start_t
                if dur < 1.0:
                    start_t = btime
                    continue

                if idx == 0:
                    if dur > 15.0:
                        lap_items.append({
                            "type": "Out Lap",
                            "lap_label": "Out Lap",
                            "lap_num": len(lap_items) + 1,
                            "start_time": start_t,
                            "end_time": btime,
                            "duration": dur,
                            "stint": 0
                        })
                elif idx == total_beacons - 1:
                    if dur > 15.0:
                        lap_items.append({
                            "type": "In Lap",
                            "lap_label": "In Lap",
                            "lap_num": len(lap_items) + 1,
                            "start_time": start_t,
                            "end_time": btime,
                            "duration": dur,
                            "stint": 0
                        })
                else:
                    if dur >= 15.0:
                        lap_num = len(lap_items) + 1
                        lap_items.append({
                            "type": "Timed",
                            "lap_label": str(lap_num),
                            "lap_num": lap_num,
                            "start_time": start_t,
                            "end_time": btime,
                            "duration": dur,
                            "stint": 0
                        })
                        if dur < fastest_dur:
                            fastest_dur = dur
                            fastest_lap = lap_num
                start_t = btime

            self.laps_info = {
                "laps": lap_items,
                "total_laps": len(lap_items),
                "fastest_lap": fastest_lap if fastest_lap is not None else 1,
                "fastest_time": fastest_dur if fastest_dur != math.inf else 0.0
            }
        elif laps_timing:
            lap_items = []
            fastest_lap = None
            fastest_dur = math.inf

            def lap_key(k):
                try:
                    return (0, int(k), "")
                except ValueError:
                    return (1, 0, str(k))

            sorted_lap_keys = sorted(laps_timing.keys(), key=lap_key)
            
            # Determine base start time and total duration using relative timestamps
            first_key = sorted_lap_keys[0]
            base_time = laps_timing[first_key]["start_time"]
            
            total_duration = 0.0
            if self.channels:
                any_ch = next(iter(self.channels.values()))
                if any_ch.messages:
                    total_duration = any_ch.messages[-1].timestamp - any_ch.messages[0].timestamp

            # Convert all lap start/end times to relative time from session start (0.0s)
            rel_laps = []
            for k in sorted_lap_keys:
                info = laps_timing[k]
                s_rel = info["start_time"] - base_time
                e_rel = info["end_time"] - base_time
                dur = e_rel - s_rel
                rel_laps.append((k, s_rel, e_rel, dur))

            # Filter valid timed laps (duration >= 20.0s)
            valid_timed = [item for item in rel_laps if item[3] >= 20.0]

            if not valid_timed:
                # Fallback: treat entire session as 1 timed lap
                lap_items.append({
                    "type": "Timed",
                    "lap_label": "1",
                    "lap_num": 1,
                    "start_time": 0.0,
                    "end_time": total_duration,
                    "duration": total_duration,
                    "stint": 0
                })
            else:
                # 1. Out Lap (if leading non-timed padding > 15.0s)
                first_start = valid_timed[0][1]
                if first_start > 15.0:
                    lap_items.append({
                        "type": "Out Lap",
                        "lap_label": "Out Lap",
                        "lap_num": 1,
                        "start_time": 0.0,
                        "end_time": first_start,
                        "duration": first_start,
                        "stint": 0
                    })

                # 2. Timed Laps
                for k, s, e, dur in valid_timed:
                    lap_num = len(lap_items) + 1
                    lap_items.append({
                        "type": "Timed",
                        "lap_label": str(k),
                        "lap_num": lap_num,
                        "start_time": s,
                        "end_time": e,
                        "duration": dur,
                        "stint": 0
                    })
                    if dur < fastest_dur:
                        fastest_dur = dur
                        fastest_lap = lap_num

                # 3. In Lap (if trailing non-timed padding > 15.0s)
                last_end = valid_timed[-1][2]
                if total_duration - last_end > 15.0:
                    in_dur = total_duration - last_end
                    lap_items.append({
                        "type": "In Lap",
                        "lap_label": "In Lap",
                        "lap_num": len(lap_items) + 1,
                        "start_time": last_end,
                        "end_time": total_duration,
                        "duration": in_dur,
                        "stint": 0
                    })

            self.laps_info = {
                "laps": lap_items,
                "total_laps": len(lap_items),
                "fastest_lap": fastest_lap if fastest_lap is not None else 1,
                "fastest_time": fastest_dur if fastest_dur != math.inf else 0.0
            }

        # Cleanup channels without any data
        empty_channels = [name for name, ch in self.channels.items() if not ch.messages]
        for name in empty_channels:
            del self.channels[name]

    def from_ibt_log(self, ibt_file_path):
        """ Creates channels populated with messages from an iRacing native .ibt binary telemetry file.

        The .ibt format consists of:
          - irsdk_header (48 bytes): ver, status, tickRate, sessionInfoUpdate,
            sessionInfoLen, sessionInfoOffset, numVars, varHeaderOffset,
            numBuf, bufLen, + 2 pad ints
          - irsdk_diskSubHeader (32 bytes at offset 48): sessionStartDate,
            sessionStartTime, sessionEndTime, lapCount, recordCount
          - irsdk_varHeader array (144 bytes each): type, offset, count,
            countAsTime, name[32], desc[64], unit[32]
          - YAML session info block (UTF-8, sessionInfoLen bytes)
          - Data buffer: n_ticks * bufLen bytes, one tick row per sample

        irsdk var types: 0=char, 1=bool, 2=int32, 3=uint32, 4=float32, 5=float64
        Reference: https://sajax.github.io/irsdkdocs/
        """
        import struct as _struct

        self.clear()
        self.laps_info = {}

        with open(ibt_file_path, "rb") as f:
            raw_file = f.read()

        if len(raw_file) < 112:
            print("ERROR: .ibt file too small to be valid")
            return

        # --- 1. Parse irsdk_header (12 ints = 48 bytes) ---
        # ver, status, tickRate, sessionInfoUpdate, sessionInfoLen, sessionInfoOffset,
        # numVars, varHeaderOffset, numBuf, bufLen, pad[2]
        (ver, status, tick_rate,
         sess_info_update, sess_info_len, sess_info_offset,
         num_vars, var_header_offset,
         num_buf, buf_len, _pad0, _pad1) = _struct.unpack_from("12i", raw_file, 0)

        # --- 2. Read bufInfo[0] (first data buffer descriptor, at byte 48) ---
        # struct irsdk_bufInfo { int tickCount; int bufOffset; int pad[2]; }
        # In .ibt disk files this is written at offset 48 directly after the 12-int header.
        buf_tick_count, buf_offset = _struct.unpack_from("2i", raw_file, 48)

        if buf_offset <= 0 or buf_offset >= len(raw_file):
            print("ERROR: Invalid .ibt data buffer offset")
            return

        n_ticks = (len(raw_file) - buf_offset) // buf_len
        if n_ticks <= 0:
            print("ERROR: .ibt file has no data ticks")
            return

        dt = 1.0 / tick_rate

        # --- 3. Parse YAML Session Info ---
        sess_yaml = raw_file[sess_info_offset: sess_info_offset + sess_info_len].decode("latin-1", errors="ignore")

        venue = ""
        driver = ""
        car = ""
        weekend_date = ""
        sess_dt = None

        for line in sess_yaml.splitlines():
            ls = line.strip()
            if ls.startswith("TrackDisplayName:"):
                venue = ls.split(":", 1)[1].strip()
            elif ls.startswith("UserName:"):
                driver = ls.split(":", 1)[1].strip()
            elif ls.startswith("CarScreenName:"):
                car = ls.split(":", 1)[1].strip()
            elif ls.startswith("WeekendDate:"):
                weekend_date = ls.split(":", 1)[1].strip()

        # Parse session datetime from YAML WeekendDate field (e.g. "2026-08-07")
        if weekend_date:
            try:
                import datetime as _dt
                sess_dt = _dt.datetime.strptime(weekend_date, "%Y-%m-%d")
            except Exception:
                pass

        self.metadata["venue_name"] = venue
        self.metadata["driver"] = driver
        self.metadata["vehicle_id"] = car
        if sess_dt:
            self.datetime = sess_dt

        # --- 4. Parse irsdk_varHeader array ---
        # Each varHeader: type(i), offset(i), count(i), countAsTime(i),
        #                 name[32s], desc[64s], unit[32s]  => 4+4+4+4+32+64+32 = 144 bytes
        _ibt_type_map = {0: np.int8, 1: np.int8, 2: np.int32, 3: np.uint32, 4: np.float32, 5: np.float64}

        var_meta = {}  # name -> (numpy_dtype, byte_offset_in_tick, count)
        for i in range(num_vars):
            base = var_header_offset + i * 144
            if base + 144 > len(raw_file):
                break
            vtype, voffset, vcount = _struct.unpack_from("3i", raw_file, base)
            vname = raw_file[base + 16: base + 48].split(b"\x00")[0].decode("latin-1")
            vunit = raw_file[base + 112: base + 144].split(b"\x00")[0].decode("latin-1")
            dtype = _ibt_type_map.get(vtype, np.float32)
            var_meta[vname] = (dtype, voffset, vcount, vunit)

        raw_data = raw_file[buf_offset: buf_offset + n_ticks * buf_len]
        times = [i * dt for i in range(n_ticks)]

        def _extract(var_name):
            if var_name not in var_meta:
                return None
            dtype, voff, vcount, vunit = var_meta[var_name]
            try:
                return np.ndarray((n_ticks,), dtype=dtype, buffer=raw_data,
                                  offset=voff, strides=(buf_len,)).astype(np.float64)
            except Exception:
                return None

        def _add_ch(ibt_name, ch_name, units, decimals, convert=None):
            arr = _extract(ibt_name)
            if arr is None:
                return
            if ibt_name == "YawNorth" and arr.max() <= 0.0:
                return
            self.add_channel(ch_name, units, float, decimals)
            vals = convert(arr) if convert else arr
            ch = self.channels[ch_name]
            for i in range(n_ticks):
                ch.messages.append(Message(times[i], float(vals[i])))

        _G = 9.80665
        _ibt_map = [
            ("Speed",              CH_GROUND_SPEED,     "km/h",    2, lambda x: x * 3.6),
            ("Lat",                CH_GPS_LATITUDE,     "deg",     7, None),
            ("Lon",                CH_GPS_LONGITUDE,    "deg",     7, None),
            ("Alt",                CH_GPS_ALTITUDE,     "m",       2, None),
            ("LatAccel",           CH_CG_ACCEL_LAT,     "G",       4, lambda x: x / _G),
            ("LongAccel",          CH_CG_ACCEL_LON,     "G",       4, lambda x: x / _G),
            ("YawRate",            CH_YAW_RATE,         "deg/s",   3, np.degrees),
            ("YawNorth",           CH_GPS_HEADING,      "deg",     2, np.degrees),
            ("SteeringWheelAngle", CH_STEERING_ANGLE,   "deg",     2, np.degrees),
            ("Throttle",           CH_THROTTLE_POS,     "%",       2, lambda x: x * 100.0),
            ("Brake",              CH_BRAKE_POS,        "%",       2, lambda x: x * 100.0),
            ("RPM",                CH_ENGINE_RPM,       "rpm",     0, None),
            ("Gear",               CH_GEAR,             "",        0, None),
            ("WaterTemp",          CH_COOLANT_TEMP,     "C",       2, None),
            ("OilTemp",            CH_ENGINE_OIL_TEMP,  "C",       2, None),
            ("OilPress",           "Engine Oil Press",  "kPa",     2, lambda x: x * 100.0),
            ("ManifoldPress",      "Manifold Press",    "kPa",     2, lambda x: x * 100.0),
            ("FuelLevel",          "Fuel Level",        "l",       2, None),
            ("Voltage",            "Battery Voltage",   "V",       2, None),
            ("TrackTemp",          "Track Temp",        "C",       2, None),
            ("LapDistPct",         "Lap Distance",      "%",       2, lambda x: x * 100.0),
        ]
        for ibt_name, ch_name, units, dec, conv in _ibt_map:
            _add_ch(ibt_name, ch_name, units, dec, conv)

        for ibt_name, ch_name in [
            ("LFspeed", "Wheel Speed FL"), ("RFspeed", "Wheel Speed FR"),
            ("LRspeed", "Wheel Speed RL"), ("RRspeed", "Wheel Speed RR"),
        ]:
            _add_ch(ibt_name, ch_name, "km/h", 2, lambda x: x * 3.6)

        for ibt_name, ch_name in [
            ("LFbrakeLinePress", "Brake Press FL"), ("RFbrakeLinePress", "Brake Press FR"),
            ("LRbrakeLinePress", "Brake Press RL"), ("RRbrakeLinePress", "Brake Press RR"),
        ]:
            _add_ch(ibt_name, ch_name, "kPa", 2, lambda x: x * 100.0)

        # --- 6. Lap detection: raw Lap counter transitions ---
        # Every forward increment of the iRacing 'Lap' counter represents a
        # real crossing of the S/F timing line (including quick-reset laps).
        # Backward transitions (e.g. 16->0) are quick-reset drops and are
        # automatically skipped because prev_lap only updates on a beacon hit.
        lap_arr = _extract("Lap")
        self.add_channel(CH_LAP_NUMBER, "", float, 0)

        if lap_arr is not None and len(lap_arr) > 1:
            beacons = []
            prev_lap = lap_arr[0]
            for i in range(1, len(lap_arr)):
                new_val = lap_arr[i]
                if new_val > prev_lap and new_val > 0:
                    beacons.append((times[i], f"Lap {int(new_val)}"))
                    prev_lap = new_val

            self.laps_info["beacons"] = beacons

            # Build lap_number channel (0 before first S/F crossing, then 1,2,3...)
            lap_nums = np.zeros(n_ticks, dtype=int)
            label_counter = 0
            prev_lap_val = lap_arr[0]
            for i in range(1, n_ticks):
                v = lap_arr[i]
                if v > prev_lap_val and v > 0:
                    label_counter += 1
                    prev_lap_val = v
                lap_nums[i] = label_counter

            for i in range(n_ticks):
                self.channels[CH_LAP_NUMBER].messages.append(Message(times[i], float(lap_nums[i])))

            # Build laps_info structure (every segment between beacons is a lap)
            lap_items = []
            total_dur = times[-1] if len(times) > 0 else 0.0
            fastest_lap = None
            fastest_dur = float("inf")

            if beacons:
                first_t = beacons[0][0]
                if first_t > 0:
                    lap_items.append({
                        "type": "Out Lap",
                        "lap_label": "Out Lap",
                        "lap_num": 1,
                        "start_time": 0.0,
                        "end_time": first_t,
                        "duration": first_t,
                        "stint": 0
                    })

                for ci in range(len(beacons) - 1):
                    s_t = beacons[ci][0]
                    e_t = beacons[ci + 1][0]
                    dur = e_t - s_t
                    lap_num = len(lap_items) + 1
                    lap_items.append({
                        "type": "Timed",
                        "lap_label": str(lap_num),
                        "lap_num": lap_num,
                        "start_time": s_t,
                        "end_time": e_t,
                        "duration": dur,
                        "stint": 0
                    })
                    if dur < fastest_dur:
                        fastest_dur = dur
                        fastest_lap = lap_num

                last_t = beacons[-1][0]
                remaining = total_dur - last_t
                if remaining > 0:
                    lap_num = len(lap_items) + 1
                    lap_items.append({
                        "type": "In Lap",
                        "lap_label": "In Lap",
                        "lap_num": lap_num,
                        "start_time": last_t,
                        "end_time": total_dur,
                        "duration": remaining,
                        "stint": 0
                    })

            self.laps_info["laps"] = lap_items
            self.laps_info["total_laps"] = len(lap_items)
            self.laps_info["fastest_lap"] = fastest_lap if fastest_lap is not None else 1
            self.laps_info["fastest_time"] = fastest_dur if fastest_dur != float("inf") else 0.0
        else:
            self.channels[CH_LAP_NUMBER].messages = [Message(times[0], 1.0)]

        # Cleanup empty channels
        empty = [name for name, ch in self.channels.items() if not ch.messages]
        for name in empty:
            del self.channels[name]

    def from_vbo_log(self, log_lines, target_lap=None):
        """ Creates channels populated with messages from a Racelogic .vbo log file. """
        self.clear()
        self.laps_info = {}

        if not log_lines:
            return

        sections = {}
        curr_sec = None
        for line in log_lines:
            line_s = line.strip()
            if line_s.startswith("[") and line_s.endswith("]"):
                curr_sec = line_s[1:-1]
                sections[curr_sec] = []
            elif curr_sec:
                sections[curr_sec].append(line_s)

        # Extract datetime from header line 1 (e.g. File created on 20/01/2026 at 07:31:48)
        if log_lines and "File created on" in log_lines[0]:
            parts = log_lines[0].strip().split()
            if len(parts) >= 6:
                d_str, t_str = parts[3], parts[5]
                try:
                    self.datetime = datetime.datetime.strptime(f"{d_str} {t_str}", "%d/%m/%Y %H:%M:%S")
                except Exception:
                    pass

        cols_raw = sections.get("column names", [""])[0].split()
        data_lines = [l for l in sections.get("data", []) if l]

        if not cols_raw or not data_lines:
            print("ERROR: Invalid VBO file, missing [column names] or [data] section")
            return

        time_idx = cols_raw.index("time") if "time" in cols_raw else -1
        if time_idx < 0:
            print("ERROR: VBO file missing 'time' column")
            return

        col_idx_map = {}
        for idx, col_name in enumerate(cols_raw):
            if col_name in CHANNEL_ALIASES:
                out_name, unit, dec = CHANNEL_ALIASES[col_name]
                fn = _parse_vbo_latlon if col_name in ("lat", "long") else float
                if out_name not in self.channels:
                    self.add_channel(out_name, unit, float, dec)
                col_idx_map[idx] = (out_name, fn)

        t0 = None
        for line_s in data_lines:
            parts = line_s.split()
            if len(parts) >= len(cols_raw):
                t_sec = _parse_vbo_time(parts[time_idx])
                if t0 is None:
                    t0 = t_sec
                t_rel = t_sec - t0
                if t_rel < 0:
                    t_rel += 86400.0  # Handle midnight wrapping
                for idx, (out_name, fn) in col_idx_map.items():
                    try:
                        v_val = fn(parts[idx])
                        self.channels[out_name].messages.append(Message(t_rel, v_val))
                    except ValueError:
                        pass

    def from_pbbuddy_log(self, log_lines, target_lap=None):
        """ Creates channels populated with messages from a PB Buddy CSV log file. """
        self.clear()
        self.laps_info = {}
        file_p = getattr(self, "log_file_path", "")

        if not log_lines:
            return

        import csv

        # 1. Scan for Data Header Line (contains >= 3 columns and starts with Time/timestamp)
        data_header_idx = -1
        for i, line in enumerate(log_lines):
            line_clean = line.strip().strip('"').strip("'")
            if not line_clean:
                continue
            try:
                parts = [p.strip().strip('"').strip("'") for p in next(csv.reader([line_clean]))]
            except Exception:
                parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]
            if len(parts) >= 3 and parts[0].lower() in ("time", "time (s)", "timestamp"):
                data_header_idx = i
                break

        if data_header_idx == -1:
            print("ERROR: Could not locate PB Buddy CSV data header line.")
            return

        # 2. Process all lines BEFORE Data Header Line as Key-Value Metadata
        for line in log_lines[:data_header_idx]:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                parts = [p.strip().strip('"').strip("'") for p in next(csv.reader([line_s]))]
            except Exception:
                parts = [p.strip().strip('"').strip("'") for p in line_s.split(",")]
            if len(parts) == 2:
                key, val = parts[0], parts[1]
                self.metadata[key] = val
                if key == "Track name":
                    self.metadata["venue_name"] = val
                elif key == "Date":
                    self.metadata["date"] = val
                elif key == "Time":
                    self.metadata["time"] = val
                elif key == "Session name":
                    self.metadata["session"] = val

        # Extract datetime from PB Buddy epoch + timezone offset or Date/Time fields
        dt = None
        if "Session start, seconds since epoch" in self.metadata:
            try:
                epoch = float(self.metadata["Session start, seconds since epoch"])
                offset_sec = 0.0
                if "Timezone offset, milliseconds" in self.metadata:
                    offset_sec = float(self.metadata["Timezone offset, milliseconds"]) / 1000.0
                dt = datetime.datetime.fromtimestamp(epoch + offset_sec, tz=datetime.timezone.utc).replace(tzinfo=None)
            except Exception:
                pass

        if dt is None:
            date_str = self.metadata.get("date") or self.metadata.get("Date") or self.metadata.get("Session start date, local timezone, YYYYMMDD")
            time_str = self.metadata.get("time") or self.metadata.get("Time")
            if date_str and time_str:
                dt_str = f"{date_str} {time_str}".strip()
                for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%y %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                    try:
                        dt = datetime.datetime.strptime(dt_str, fmt)
                        break
                    except Exception:
                        pass

        if dt:
            self.datetime = dt

        # 3. Read Headers and Units Row
        header_line = log_lines[data_header_idx].strip()
        headers = [h.strip().strip('"').strip("'") for h in header_line.split(",")]

        units = []
        if data_header_idx + 1 < len(log_lines):
            next_line = log_lines[data_header_idx + 1].strip()
            unit_parts = [u.strip().strip('"').strip("'") for u in next_line.split(",")]
            if unit_parts and unit_parts[0].lower() in ("sec", "s", "seconds", "m/s", "deg", "m"):
                units = unit_parts

        mapping = {
            "Time": ("Time", "s", 3),
            "GPS Latitude": ("GPS Latitude", "deg", 7),
            "GPS Longitude": ("GPS Longitude", "deg", 7),
            "GPS Speed": ("Ground Speed", "km/h", 5),  # Convert m/s -> km/h
            "GPS Heading": ("GPS Heading", "deg", 3),
            "GPS Altitude": ("GPS Altitude", "m", 3),
            # Lap count column - added by PB Buddy on request (Timur, 2026-08)
            "Lap Count": ("Lap Number", "", 0),
            "Lap": ("Lap Number", "", 0),
            "Lap Number": ("Lap Number", "", 0),
        }

        chan_indices = {}
        for idx, h in enumerate(headers):
            if h in mapping:
                out_name, out_unit, out_dec = mapping[h]
                self.add_channel(out_name, out_unit, float, out_dec)
                chan_indices[idx] = (h, out_name)
            elif h != "Time":
                unit = units[idx] if idx < len(units) else ""
                self.add_channel(h, unit, float, 3)
                chan_indices[idx] = (h, h)

        # 4. Parse Data Rows
        start_data_idx = data_header_idx + (2 if units else 1)
        for line in log_lines[start_data_idx:]:
            line_s = line.strip()
            if not line_s or line_s.startswith("//") or line_s.startswith("#"):
                continue
            parts = line_s.split(",")
            if len(parts) >= len(headers):
                try:
                    t = float(parts[0])
                    for idx, (orig_name, out_name) in chan_indices.items():
                        val = float(parts[idx])
                        if orig_name == "GPS Speed":
                            val = val * 3.6  # m/s -> km/h
                        self.channels[out_name].messages.append(Message(t, val))
                except ValueError:
                    continue

    def from_rcz_log(self, rcz_file_path, target_lap=None, target_stint=None, min_lap_sec=15.0):
        """ Creates channels populated with messages directly from a RaceChrono .rcz archive.

        rcz_file_path: Path to the .rcz file
        target_lap: String, int, or None. If None or 'all', processes all laps in the session.
        """
        import zipfile
        import json
        import os

        self.clear()
        self.laps_info = {}

        if not os.path.isfile(rcz_file_path):
            print("ERROR: RCZ file %s does not exist" % rcz_file_path)
            return

        with zipfile.ZipFile(rcz_file_path, 'r') as z:
            all_names = z.namelist()
            prefix = ""
            if target_stint is not None and str(target_stint).lower() != "all":
                stint_value = int(target_stint)
                prefix = "resume_%s/" % stint_value if stint_value > 0 else ""
            namelist = [n[len(prefix):] for n in all_names if n.startswith(prefix)]
            def read_channel(name):
                return z.read(prefix + name)

            # RCZ binary channel naming conventions (RaceChrono internal format).
            # Device numbers encode the sensor type:
            #   1/200 = external GPS (RaceBox/VBOX),    1/100 = phone GPS
            #   2/201 = dedicated accelerometer
            #   3/202 = gyroscope
            #   4/101 = phone IMU / secondary OBD
            #   12/100 = primary OBD (car ECU via CAN)
            _GPS_DEV_PFX      = ("channel_1_200_0_", "channel_1_100_0_")
            _GPS_TS_KEYS      = ("channel_1_200_0_1_1", "channel_1_200_0_2_1")
            _ACCEL_LAT        = "channel_2_201_0_10_0"
            _ACCEL_LONG       = "channel_2_201_0_9_0"
            _ACCEL_Z          = "channel_2_201_0_11_0"
            _GYRO_Z           = "channel_3_202_0_14_0"
            _GYRO_X           = "channel_3_202_0_12_0"
            _OBD_DEV12_PFX    = "channel_12_100_"
            _OBD_DEV4_TS_KEY  = "channel_4_101_0_1_1"

            if "session.json" not in all_names:
                print("ERROR: Invalid RCZ file, missing session.json")
                return

            session_json = json.loads(z.read("session.json").decode("utf-8"))
            first_t = session_json.get("firstTimestamp", 0)

            if "trackId.json" in all_names:
                try:
                    track_json = json.loads(z.read("trackId.json").decode("utf-8"))
                    traps_list = track_json.get("track", {}).get("traps", [])
                    for t in traps_list:
                        if "centerLatitude" in t and "centerLongitude" in t:
                            self.traps.append({
                                "name": t.get("name", "Split"),
                                "lat": t["centerLatitude"] / 6000000.0,
                                "lon": t["centerLongitude"] / 6000000.0,
                                "type": t.get("type", 4)
                            })
                except Exception:
                    pass

            ts_ms = session_json.get("timeCreated") or session_json.get("firstTimestamp")
            if ts_ms and ts_ms > 1e8:
                import datetime
                self.datetime = datetime.datetime.fromtimestamp(ts_ms / 1000.0)
            else:
                self.datetime = self._extract_datetime_from_text([], rcz_file_path)

            # Store metadata
            self.rcz_metadata = {
                "title": session_json.get("title", ""),
                "trackName": session_json.get("trackName", ""),
                "timeCreated": session_json.get("timeCreated", 0)
            }
            self._extract_metadata([], rcz_file_path)
            if session_json.get("trackName"):
                self.metadata["venue_name"] = session_json.get("trackName")
            if session_json.get("title"):
                self.metadata["event_session"] = session_json.get("title")
            notes = session_json.get("notes") or session_json.get("description")
            if notes:
                self.metadata["short_comment"] = notes

            # Determine active stint to process
            active_stint = 0
            if target_stint is not None and str(target_stint).lower() != "all":
                try:
                    active_stint = int(target_stint)
                except ValueError:
                    pass
            time_file = None
            for candidate in _GPS_TS_KEYS:
                if candidate in namelist:
                    time_file = candidate
                    break

            if not time_file:
                for name in namelist:
                    if name.startswith("channel_") and name.endswith("_1_1"):
                        time_file = name
                        break

            if not time_file:
                print("ERROR: Could not find timestamp channel in RCZ file")
                return

            t_data = read_channel(time_file)
            timestamps_ms = np.frombuffer(t_data, dtype="<i8")
            uptimes_ms = np.frombuffer(t_data, dtype="<i4")[::2].astype(np.float64)
            stint_uptime_start = uptimes_ms[0] if len(uptimes_ms) else 0.0

            if len(timestamps_ms) > 0 and timestamps_ms[0] > 1e8:
                import datetime
                self.datetime = datetime.datetime.fromtimestamp(timestamps_ms[0] / 1000.0)

            times_sec = (timestamps_ms - first_t) / 1000.0
            # Normalize RCZ time so every exported MoTeC session starts at zero.
            time_origin = float(times_sec[0]) if len(times_sec) else 0.0
            times_sec = times_sec - time_origin
            n_samples = len(times_sec)

            # Map laps from session.json
            laps_raw = session_json.get("laps", [])
            
            # Group raw laps by stint
            stints_map = {}
            for l in laps_raw:
                st_id = int(l.get("sessionResume", 0))
                if st_id not in stints_map:
                    stints_map[st_id] = []
                stints_map[st_id].append(l)

            stint_laps = stints_map.get(active_stint, [])

            # Reconstruct clean lap structure for this stint by filtering laps that overlap with binary recording range
            stint_duration_s = times_sec[-1] if len(times_sec) else 0.0
            stint_start_ms = timestamps_ms[0] if len(timestamps_ms) else first_t
            stint_end_ms = timestamps_ms[-1] if len(timestamps_ms) else (first_t + int(stint_duration_s * 1000))

            # Filter raw laps that actually overlap with binary recording range
            overlapping_laps = []
            for l in stint_laps:
                s_ms = l.get("startTimestamp")
                f_ms = l.get("finishTimestamp")
                if not s_ms:
                    continue
                cmp_end = f_ms if f_ms is not None else stint_end_ms
                if s_ms < stint_end_ms and cmp_end > stint_start_ms:
                    overlapping_laps.append(l)

            reconstructed_laps = []

            # 1. Out Lap (if leading non-timed padding > 15.0s)
            first_timed_s = (overlapping_laps[0].get("startTimestamp") - stint_start_ms) / 1000.0 if overlapping_laps else 0.0
            if first_timed_s > 15.0:
                reconstructed_laps.append({
                    "type": "Out Lap",
                    "lap_label": "Out Lap",
                    "lap_num": 1,
                    "start_time": 0.0,
                    "end_time": first_timed_s,
                    "duration": first_timed_s,
                    "stint": active_stint
                })

            # 2. Timed Laps & In Lap inside recording window
            for l in overlapping_laps:
                s_ms = l.get("startTimestamp")
                f_ms = l.get("finishTimestamp")
                orig_num = l.get("number")

                start_s = max(0.0, (s_ms - stint_start_ms) / 1000.0)

                if f_ms is not None:
                    end_s = min(stint_duration_s, (f_ms - stint_start_ms) / 1000.0)
                    dur_s = end_s - start_s
                    if dur_s >= 15.0:  # Must be at least 15s to be a valid lap
                        reconstructed_laps.append({
                            "type": "Timed",
                            "lap_label": str(len(reconstructed_laps) + 1),
                            "lap_num": len(reconstructed_laps) + 1,
                            "start_time": start_s,
                            "end_time": end_s,
                            "duration": dur_s,
                            "stint": active_stint,
                            "orig_num": orig_num
                        })
                else:
                    end_s = stint_duration_s
                    dur_s = end_s - start_s
                    if dur_s >= 15.0:
                        reconstructed_laps.append({
                            "type": "In Lap",
                            "lap_label": "In Lap",
                            "lap_num": len(reconstructed_laps) + 1,
                            "start_time": start_s,
                            "end_time": end_s,
                            "duration": dur_s,
                            "stint": active_stint,
                            "orig_num": orig_num
                        })

            # 3. Trailing In Lap check (if trailing non-timed padding > 15.0s)
            if overlapping_laps and overlapping_laps[-1].get("finishTimestamp") is not None:
                last_finish_ms = overlapping_laps[-1].get("finishTimestamp")
                in_dur_s = (stint_end_ms - last_finish_ms) / 1000.0
                if in_dur_s > 15.0:
                    reconstructed_laps.append({
                        "type": "In Lap",
                        "lap_label": "In Lap",
                        "lap_num": len(reconstructed_laps) + 1,
                        "start_time": (last_finish_ms - stint_start_ms) / 1000.0,
                        "end_time": stint_duration_s,
                        "duration": in_dur_s,
                        "stint": active_stint
                    })

            # Build lap_numbers array for data logging
            lap_numbers = np.ones(n_samples, dtype=int)
            for r in reconstructed_laps:
                m = (times_sec >= r["start_time"]) & (times_sec <= r["end_time"])
                lap_numbers[m] = r["lap_num"]

            # Filter target_lap if specified
            target_lap_int = None
            if target_lap is not None and str(target_lap).lower() != "all":
                try:
                    target_lap_int = int(target_lap)
                except ValueError:
                    pass

            if target_lap_int is not None:
                mask = (lap_numbers == target_lap_int)
            else:
                mask = np.ones(n_samples, dtype=bool)

            sample_times = times_sec[mask]
            if len(sample_times) > 0 and len(timestamps_ms) > 0 and timestamps_ms[0] > 1e8:
                actual_start_ms = timestamps_ms[0] + (sample_times[0] * 1000.0)
                import datetime
                self.datetime = datetime.datetime.fromtimestamp(actual_start_ms / 1000.0)

            # Store laps_info for ldx export
            fastest_lap_num = 1
            fastest_dur = float("inf")
            for r in reconstructed_laps:
                if r["type"] == "Timed" and r["duration"] < fastest_dur:
                    fastest_dur = r["duration"]
                    fastest_lap_num = r["lap_num"]

            self.laps_info = {
                "laps": reconstructed_laps,
                "total_laps": len(reconstructed_laps),
                "fastest_lap": fastest_lap_num,
                "fastest_time": fastest_dur if fastest_dur != float("inf") else 0.0,
                "session_duration": stint_duration_s,
            }
            
            # Helper to add channel messages
            def populate_channel(name, units, values_array, decimals=2):
                if len(values_array) < n_samples:
                    return
                values_array = values_array[:n_samples]
                filtered_vals = values_array[mask]
                self.add_channel(name, units, float, decimals)
                self.channels[name].decimals = decimals
                self.channels[name].messages = [Message(sample_times[i], filtered_vals[i]) for i in range(len(sample_times))]

            # Lap Number Channel
            populate_channel("Lap Number", "", lap_numbers, 0)
            # Running Time Channel
            populate_channel("Running Time", "s", times_sec, 2)

            # Auto-detect GPS device prefix (type 200 = external VBOX, type 100 = phone GPS)
            gps_prefix = None
            for _pfx in _GPS_DEV_PFX:
                if any(n.startswith(_pfx) for n in namelist):
                    gps_prefix = _pfx
                    break

            if gps_prefix:
                # 2. Parse Speed
                _spd_key = gps_prefix + "4_0"
                if _spd_key in namelist:
                    raw_spd = np.frombuffer(read_channel(_spd_key), dtype="<i4")
                    if len(raw_spd) >= n_samples:
                        populate_channel("Ground Speed", "km/h", (raw_spd / 1000.0) * 3.6, 2)

                # 3. Parse Latitude & Longitude
                _ll_key = gps_prefix + "3_1"
                if _ll_key in namelist:
                    raw_ll = np.frombuffer(read_channel(_ll_key), dtype="<i4")
                    if len(raw_ll) >= n_samples * 2:
                        raw_ll = raw_ll[:n_samples * 2].reshape(-1, 2)
                        populate_channel("GPS Latitude", "deg", raw_ll[:, 0] / 6000000.0, 7)
                        populate_channel("GPS Longitude", "deg", raw_ll[:, 1] / 6000000.0, 7)

                # 4. Parse Altitude
                _alt_key = gps_prefix + "5_0"
                if _alt_key in namelist:
                    raw_alt = np.frombuffer(read_channel(_alt_key), dtype="<i4")
                    if len(raw_alt) >= n_samples:
                        populate_channel("GPS Altitude", "m", raw_alt / 1000.0)

                # 5. Parse GPS Heading
                _hdg_key = gps_prefix + "6_0"
                if _hdg_key in namelist:
                    raw_hdg = np.frombuffer(read_channel(_hdg_key), dtype="<i4")
                    if len(raw_hdg) >= n_samples:
                        populate_channel("GPS Heading", "deg", raw_hdg / 1000.0)

            # 6. Parse Accelerations (device 2, type 201)
            # If a per-device timestamp file exists, resample onto GPS time grid using
            # np.interp.  This corrects for slight rate drift (e.g. IMU at 25.008 Hz vs
            # GPS at 25.000 Hz) which accumulates to 48-sample / 1.9 s error over a
            # 1115 s session, causing a measurable lag in the exported data.
            _IMU_TS_KEY  = "channel_2_201_0_1_1"
            _GYRO_TS_KEY = "channel_3_202_0_1_1"

            def _imu_times(ts_key):
                """Return relative time array (seconds, vs stint_uptime_start) for a device ts file."""
                if ts_key not in namelist:
                    return None
                raw = np.frombuffer(read_channel(ts_key), dtype="<i4")
                return (raw[::2].astype(np.float64) - stint_uptime_start) / 1000.0

            def _parse_imu_channel(ch_key, out_name, units, scale, decimals=2, ts_key=_IMU_TS_KEY):
                if ch_key not in namelist:
                    return
                raw = np.frombuffer(read_channel(ch_key), dtype="<i4").astype(np.float64) * scale
                imu_t = _imu_times(ts_key)
                if imu_t is not None and len(imu_t) == len(raw):
                    # Resample via timestamps - handles rate drift and small offsets
                    resampled = np.interp(times_sec, imu_t, raw)
                    populate_channel(out_name, units, resampled, decimals)
                elif len(raw) >= n_samples:
                    # Fallback: naive truncation (off by  1 sample)
                    populate_channel(out_name, units, raw, decimals)

            _parse_imu_channel(_ACCEL_LAT,  "CG Accel Lateral",     "G",     -1.0 / 10000.0)
            _parse_imu_channel(_ACCEL_LONG, "CG Accel Longitudinal", "G",     1.0 / 10000.0)
            _parse_imu_channel(_ACCEL_Z,    "Lean Angle",            "deg",   1.0 / 10000.0)

            # 7. Parse Gyroscope / Yaw Rate (device 3, type 202)
            if _GYRO_Z in namelist:
                _parse_imu_channel(_GYRO_Z, "Chassis Yaw Rate", "deg/s",
                                   -1.0 / 1000.0, ts_key=_GYRO_TS_KEY)
            elif _GYRO_X in namelist:
                _parse_imu_channel(_GYRO_X, "x_rate_of_rotation", "",
                                   1.0 / 1000.0, ts_key=_GYRO_TS_KEY)

            # 8. Parse OBD-II / CAN Channels
            # RCZ stores binary channel values as contiguous IEEE 754 float64 (double precision) values.
            # Values are ALREADY in engineering units (1:1 scale).
            # Internal RCZ channel PID to MoTeC channel name & unit mapping:
            rcz_pid_map = {
                "10024": ("Engine RPM", "rpm", 1.0, 0.0),            # 605 ~ 7489 rpm
                "10025": ("Accelerator Pos", "%", 1.0, 0.0),         # 0 ~ 100 % (Pedal)
                "10071": ("Throttle Pos", "%", 1.0, 0.0),            # 0 ~ 100 % (Throttle Valve)
                "1002":  ("Brake Pos", "%", 1.0, 0.0),               # 0 ~ 100 %
                "1033":  ("Brake Press", "kPa", 1.0, 0.0),           # 0 ~ 9600 kPa
                "1007":  ("Engine Oil Press", "kPa", 1.0, 0.0),      # 110 ~ 683 kPa
                "10066": ("Engine Oil Temp", "C", 1.0, 0.0),         # 57 ~ 104 C
                "10026": ("Coolant Temp", "C", 1.0, 0.0),            # 67 ~ 95 C
                "1005":  ("Gearbox Temp", "C", 1.0, 0.0),            # 63 ~ 114 C
                "10029": ("Intake Temp", "C", 1.0, 0.0),             # 29 ~ 42 C
                "1001":  ("Steering Angle", "deg", -1.0, 0.0),        # -294 ~ 457 deg
                "1004":  ("Gear", "", 1.0, 0.0),                     # -1 ~ 5 (Integer gear)
                "4":     ("Ground Speed", "km/h", 3.6, 0.0),         # raw is m/s (0 ~ 50 m/s -> km/h)
                "51":    ("Chassis Yaw Rate", "deg/s", 1.0, 0.0),    # -37 ~ 39 deg/s
                "7":     ("Roll Angle", "deg", 1.0, 0.0),            # Roll Angle (-11.6 ~ 11.2 deg)
                "8":     ("Pitch Angle", "deg", 1.0, 0.0),           # Pitch Angle (-10.6 ~ 4.0 deg)
                "10031": ("Ambient Temp", "C", 1.0, 0.0),            # Ambient Temp (16 ~ 87 C)
                "1053576": ("Wheel Speed Avg", "rpm", 1.0, 0.0),     # Wheel Speed (1296 ~ 3230)
            }

            for name in namelist:
                base_fname = os.path.basename(name)
                if not base_fname.startswith(_OBD_DEV12_PFX) or not base_fname.endswith("_1_1"):
                    continue
                parts = base_fname.split("_")
                pid = parts[3]
                raw_time_data = np.frombuffer(read_channel(name), dtype="<i4")
                dir_prefix = os.path.dirname(name)
                companion_fname = f"channel2_12_100_{pid}_{pid}_3"
                companion = (dir_prefix + "/" + companion_fname) if dir_prefix else companion_fname
                if companion not in namelist or len(raw_time_data) < 2 or len(raw_time_data) % 2:
                    continue
                raw_times = raw_time_data[::2].astype(np.float64)
                # Correct binary format: float64 (double precision, 8 bytes per value)
                value_data = np.frombuffer(read_channel(companion), dtype="<f8")
                count = min(len(raw_times), len(value_data))
                if count < 2:
                    continue
                raw_times = raw_times[:count]
                raw_values = value_data[:count]
                rel_times = (raw_times - stint_uptime_start) / 1000.0
                if rel_times[-1] <= 0:
                    continue
                if pid == "1004":
                    values = _interp_zoh(times_sec, rel_times, raw_values)
                else:
                    values = np.interp(times_sec, rel_times, raw_values)
                if pid in rcz_pid_map:
                    ch_name, ch_unit, ch_scale, ch_offset = rcz_pid_map[pid]
                    if ch_name not in self.channels:
                        vals_processed = values * ch_scale + ch_offset
                        if pid == "1004":
                            vals_processed = np.round(vals_processed).clip(-1, 6)
                        populate_channel(ch_name, ch_unit, vals_processed)
                else:
                    populate_channel(f"OBD_{pid}", "", values)

            # 8b. Parse OBD-II / CAN Channels - Device 4, Type 101 (shared timestamp format)
            # Format: channel_4_101_0_1_1 = shared i32[::2] uptime timestamps for all PIDs
            #         channel2_4_101_0_{pid}_3 = per-PID float64 values (same length as timestamps)
            # Device 4/type 101 uses phone-IMU channels for PIDs 7/8/49/50 - different from
            # the GR86-ECU meaning of the same PIDs on device 12/type 100.
            _G = 9.80665
            rcz_dev4_pid_overrides = {
                # PID 7 on device 4 = phone lateral accelerometer (m/s^2), NOT ECU Roll Angle
                "7":  ("CG Accel Lateral",     "G",    1.0 / _G, 0.0),
                # PID 8 on device 4 = phone vertical accelerometer (m/s^2, includes gravity) - skip
                "8":  None,
            }
            dev4_ts_key = _OBD_DEV4_TS_KEY
            dir_prefix = os.path.dirname(dev4_ts_key)
            if any(os.path.basename(n) == os.path.basename(dev4_ts_key) and
                   os.path.dirname(n) == dir_prefix for n in namelist):
                actual_key = next(
                    n for n in namelist
                    if os.path.basename(n) == os.path.basename(_OBD_DEV4_TS_KEY)
                )
                raw_obd4_ts = np.frombuffer(read_channel(actual_key), dtype="<i4")
                if len(raw_obd4_ts) >= 2 and len(raw_obd4_ts) % 2 == 0:
                    obd4_uptimes = raw_obd4_ts[::2].astype(np.float64)
                    obd4_rel_times = (obd4_uptimes - stint_uptime_start) / 1000.0
                    # Merge standard map with device-4-specific overrides
                    dev4_map = {**rcz_pid_map, **{
                        k: v for k, v in rcz_dev4_pid_overrides.items() if v is not None
                    }}
                    for pid in set(list(rcz_pid_map.keys()) + list(rcz_dev4_pid_overrides.keys())):
                        override = rcz_dev4_pid_overrides.get(pid, "NOT_OVERRIDDEN")
                        if override is None:
                            continue  # explicitly skipped for device 4
                        mapping = dev4_map.get(pid)
                        if mapping is None:
                            continue
                        ch_name, ch_unit, ch_scale, ch_offset = mapping
                        # Skip PIDs that use the GPS speed channel (already parsed above)
                        if pid == "4":
                            continue
                        val_fname = f"channel2_4_101_0_{pid}_3"
                        val_key = next(
                            (n for n in namelist if os.path.basename(n) == val_fname),
                            None,
                        )
                        if val_key is None:
                            continue
                        value_data = np.frombuffer(read_channel(val_key), dtype="<f8")
                        count = min(len(obd4_uptimes), len(value_data))
                        if count < 2:
                            continue
                        rel_t = obd4_rel_times[:count]
                        vals = value_data[:count]
                        if pid == "1004":
                            interpolated = _interp_zoh(times_sec, rel_t, vals)
                        else:
                            interpolated = np.interp(times_sec, rel_t, vals)
                        processed = interpolated * ch_scale + ch_offset
                        if pid == "1004":
                            processed = np.round(processed).clip(-1, 6)
                        if ch_name not in self.channels:
                            populate_channel(ch_name, ch_unit, processed)

        # Fallback for Chassis Yaw Rate if missing
        self._derive_yaw_rate_from_gps_heading()

    def from_accessport_log(self, log_lines):
        """ Creates channels populated with messages from a COBB Accessport CSV log file.

        This will create a channel for each column in the CSV file, with the name and units of that
        channel taken from the CSV header. Any non numeric data will be ignored, and that channel
        will be removed.

        log_lines: List, containing CSV log lines
        """

        self.from_csv_log(log_lines)

        # Accessport logs have a column for AP info which is not of any value so we'll delete it
        for key in self.channels.keys():
            if "AP Info" in key:
                del self.channels[key]
                break

        # Update all the channel names and units
        for channel_name, channel in self.channels.items():
            # Channels have the format "Name (Units)"
            name, units = channel_name.split(" (")
            units = units[:-1]

            channel.name = name
            channel.units = units

    @staticmethod
    def __parse_can_log_line(line):
        try:
            stamp, bus, msg = line.split()
            stamp = float(stamp[1:-1])
            can_id, data = msg.split("#")
            can_id = int(can_id, 16)
            data = bytearray.fromhex(data)
            return stamp, bus, can_id, data
        except (ValueError, IndexError):
            return None, None, None, None

    def __str__(self):
        output = "Log: %s, Duration: %f s" % (self.name, (self.end() - self.start()))
        for channel_name, channel_data in self.channels.items():
            output += "\n\t%s" % channel_data
        return output

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
        """ Computes the average frequency from the samples based on the duration of the channel
        and the number of messages"""
        if len(self.messages) >= 2:
            dt = self.end() - self.start()
            return len(self.messages) / dt
        else:
            return 0

    def resample(self, start_time, end_time, frequency):
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

        self.messages = [Message(float(new_t[i]), float(new_v[i])) for i in range(num_msgs)]

    def __str__(self):
        return "Channel: %s, Units: %s, Decimals: %d, Messages: %d, Frequency: %.2f Hz" % \
        (self.name, self.units, self.decimals, len(self.messages), self.avg_frequency())

class Message(object):
    """ A single message in a time series of data. """
    def __init__(self, timestamp: float = 0.0, value: float = 0.0):
        self.timestamp = float(timestamp)
        self.value = float(value)

    def __str__(self):
        return "t=%f, value=%f" % (self.timestamp, self.value)
