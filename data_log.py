from __future__ import annotations

import math

import numpy as np


class DataLog(object):
    channels: dict[str, Channel]
    """ Container for storing log data which contains a set of channels with time series data."""
    def __init__(self, name=""):
        self.name = name
        self.channels = {}
        self.datetime = None
        self.metadata = {}

    def clear(self):
        self.channels = {}
        self.datetime = None
        self.metadata = {}

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

        if t != math.inf:
            return t
        else:
            return 0.0

    def end(self):
        """ Returns the latest timestamp from all existing channels [s]. """
        end = 0
        for name, channel in self.channels.items():
            end = max(end, channel.end())

        return end

    def duration(self):
        """ Returns the duration of the log [s]. """
        return self.end() - self.start()

    def resample(self, frequency):
        """ Resamples all channels such that all messages occur at a fixed frequency.

        See the resample method of the Channel class for more details.
        """
        start = self.start()
        end = self.end()
        for channel_name in self.channels:
            self.channels[channel_name].resample(start, end, frequency)

    def calculate_math_channels(self):
        """ Calculates advanced racing math channels to match simulator-level data richness.

        This should be called after resampling the log to a fixed frequency.
        Generates: Tire Slip Angles (FL/FR/RL/RR), G-Sum, Understeer Index, and Input Rates.
        Constants tuned for Toyota GR86 (Wheelbase: 2.575m, Ratio: 13.5:1).
        """
        # Fallback: derive CG Accel Lateral from kinematic (Vx * YawRate) if missing
        if "CG Accel Lateral" not in self.channels and "Ground Speed" in self.channels and "Chassis Yaw Rate" in self.channels:
            spd_chan = self.channels["Ground Speed"]
            yaw_chan = self.channels["Chassis Yaw Rate"]
            n_m = len(spd_chan.messages)
            if n_m > 0:
                t_arr = np.array([m.timestamp for m in spd_chan.messages])
                vx_ms = np.array([m.value / 3.6 if spd_chan.units == "km/h" else m.value for m in spd_chan.messages])
                yr_rad = np.radians([m.value for m in yaw_chan.messages])
                ay_derived = (vx_ms * yr_rad) / 9.80665
                self.add_channel("CG Accel Lateral", "G", float, 2)
                self.channels["CG Accel Lateral"].messages = [Message(t_arr[i], ay_derived[i]) for i in range(n_m)]

        # Fallback: derive CG Accel Longitudinal from speed derivative if missing
        if "CG Accel Longitudinal" not in self.channels and "Ground Speed" in self.channels:
            spd_chan = self.channels["Ground Speed"]
            n_m = len(spd_chan.messages)
            if n_m >= 2:
                t_arr = np.array([m.timestamp for m in spd_chan.messages])
                vx_ms = np.array([m.value / 3.6 if spd_chan.units == "km/h" else m.value for m in spd_chan.messages])
                ax_derived = np.gradient(vx_ms, t_arr) / 9.80665
                self.add_channel("CG Accel Longitudinal", "G", float, 2)
                self.channels["CG Accel Longitudinal"].messages = [Message(t_arr[i], ax_derived[i]) for i in range(n_m)]

        # --- Kinematic Vehicle Dynamics (Body Slip, Tire Slip, Understeer Index) ---
        has_kinematics = all(req in self.channels for req in ["Ground Speed", "CG Accel Lateral", "Chassis Yaw Rate"])
        if has_kinematics:
            vx_chan = self.channels["Ground Speed"]
            n = len(vx_chan.messages)
            if n >= 2:
                time = np.array([m.timestamp for m in vx_chan.messages])

                # Unit detection for speed (ensure m/s for internal math)
                vx = np.array([m.value for m in vx_chan.messages])
                if vx_chan.units == "km/h":
                    vx /= 3.6

                ay = np.array([m.value * 9.80665 for m in self.channels["CG Accel Lateral"].messages])

                # Chassis Yaw Rate was flipped in from_racechrono_log (* -1.0)
                # Flip it back to physical convention (Positive = Left Turn) for kinematics
                yaw_rate_degs = np.array([m.value for m in self.channels["Chassis Yaw Rate"].messages])
                yaw_rate = np.radians(yaw_rate_degs * -1.0)

                dt = np.zeros(n)
                dt[1:] = np.diff(time)

                # --- 1. Body Slip Angle (Beta) Calculation ---
                vy = np.zeros(n)
                beta = np.zeros(n)
                tau = 2.0

                for i in range(1, n):
                    vy_dot = ay[i] - (vx[i] * yaw_rate[i])
                    alpha = np.exp(-dt[i] / tau)
                    vy[i] = (vy[i-1] + vy_dot * dt[i]) * alpha

                    is_straight = (abs(ay[i]) < 0.49) and (abs(yaw_rate_degs[i]) < 1.0)
                    if is_straight:
                        vy[i] = 0.0

                    if vx[i] > 5.0:
                        beta[i] = np.arctan2(vy[i], vx[i])
                    else:
                        beta[i] = 0.0

                # --- 2. Tire Slip Angle Calculation (Kinematic Bicycle Model) ---
                ratio = 13.5
                wheelbase = 2.575
                lf = 1.25   # Distance from CG to front axle
                lr = 1.325  # Distance from CG to rear axle

                slip_f = np.zeros(n)
                slip_r = np.zeros(n)
                steer_rad = np.zeros(n)

                if "Steering Angle" in self.channels:
                    steer_deg = np.array([m.value for m in self.channels["Steering Angle"].messages])
                    steer_rad = np.radians(steer_deg / ratio)

                for i in range(n):
                    if vx[i] > 5.0:
                        slip_f[i] = np.degrees(steer_rad[i] - np.arctan2(vy[i] + yaw_rate[i]*lf, vx[i]))
                        slip_r[i] = np.degrees(-np.arctan2(vy[i] - yaw_rate[i]*lr, vx[i]))

                tire_names = ["Tire Slip Angle FL", "Tire Slip Angle FR", "Tire Slip Angle RL", "Tire Slip Angle RR"]
                for name in tire_names:
                    self.add_channel(name, "deg", float, 2)
                    src_data = slip_f if "F" in name else slip_r
                    self.channels[name].messages = [Message(time[i], src_data[i]) for i in range(n)]

                # --- 3. Understeer Index ---
                us_index = np.zeros(n)
                for i in range(n):
                    if vx[i] > 5.0:
                        us_index[i] = np.degrees(steer_rad[i]) - np.degrees(wheelbase * yaw_rate[i] / vx[i])

                self.add_channel("Understeer Index", "deg", float, 2)
                self.channels["Understeer Index"].messages = [Message(time[i], us_index[i]) for i in range(n)]

        # --- 4. G Force Combined (G-Sum) ---
        if "CG Accel Longitudinal" in self.channels and "CG Accel Lateral" in self.channels:
            ax_msgs = self.channels["CG Accel Longitudinal"].messages
            ay_msgs = self.channels["CG Accel Lateral"].messages
            n_g = min(len(ax_msgs), len(ay_msgs))
            if n_g >= 2:
                time_g = [ax_msgs[i].timestamp for i in range(n_g)]
                ax = np.array([ax_msgs[i].value for i in range(n_g)])
                ay_g = np.array([ay_msgs[i].value for i in range(n_g)])
                g_sum = np.sqrt(ax**2 + ay_g**2)
                self.add_channel("G Force Combined", "G", float, 2)
                self.channels["G Force Combined"].messages = [Message(time_g[i], g_sum[i]) for i in range(n_g)]

        # --- Derive Brake Pos from Brake Press if Brake Pos is missing (5/29 algorithm: 96 kPa = 1% Brake Pos) ---
        if "Brake Press" in self.channels and "Brake Pos" not in self.channels:
            press_chan = self.channels["Brake Press"]
            n_p = len(press_chan.messages)
            if n_p > 0:
                time_p = [m.timestamp for m in press_chan.messages]
                press_vals = np.array([m.value for m in press_chan.messages])
                if press_chan.units == "bar":
                    press_vals *= 100.0
                bpos_derived = np.clip(press_vals / 96.0, 0.0, 100.0)
                self.add_channel("Brake Pos", "%", float, 2)
                self.channels["Brake Pos"].messages = [Message(time_p[i], bpos_derived[i]) for i in range(n_p)]

        # --- 5. Input Rates (Derivatives) ---
        self.__calculate_rate("Steering Angle", "deg/s")
        self.__calculate_rate("Throttle Pos", "%/s")
        self.__calculate_rate("Brake Pos", "%/s")

        # --- 6. Dual GPS Channel Mirroring for MoTeC Workspace Compatibility ---
        if "GPS Latitude" in self.channels and "Real GPS Latitude" not in self.channels:
            lat_chan = self.channels["GPS Latitude"]
            self.add_channel("Real GPS Latitude", "deg", float, 7)
            self.channels["Real GPS Latitude"].messages = [Message(m.timestamp, m.value) for m in lat_chan.messages]

        if "GPS Longitude" in self.channels and "Real GPS Longitude" not in self.channels:
            lon_chan = self.channels["GPS Longitude"]
            self.add_channel("Real GPS Longitude", "deg", float, 7)
            self.channels["Real GPS Longitude"].messages = [Message(m.timestamp, m.value) for m in lon_chan.messages]

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
                    elif k in ["session", "segment"]:
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

        if not log_lines:
            return

        # Get the channel names, ignore the first column as it is assumed to be time
        header = log_lines[0].strip("\n")
        channel_names = [name.strip().strip('"').strip("'") for name in header.split(",")[1:]]

        # We'll keep a map of names and column numbers for easy channel lookups when parsing rows
        i = 0
        channel_dict = {}
        for name in channel_names:
            self.add_channel(name, "", float, 0)

            channel_dict[name] = i
            i += 1

        # Go through each line grabbing all the channel values
        for line in log_lines[1:]:
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

            raw_lower = raw_name.lower().strip()
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

        # If Chassis Yaw Rate is still missing, fallback to calculating it from GPS Heading derivative
        if "Chassis Yaw Rate" not in self.channels and "GPS Heading" in self.channels:
            gps_h_chan = self.channels["GPS Heading"]
            if len(gps_h_chan.messages) >= 2:
                times = np.array([m.timestamp for m in gps_h_chan.messages])
                headings = np.array([m.value for m in gps_h_chan.messages])
                dt = np.diff(times)
                dt[dt == 0] = 0.001
                h_unwrapped = np.unwrap(np.radians(headings))
                h_deg = np.degrees(h_unwrapped)
                yaw_rate_val = -np.gradient(h_deg, times)
                
                self.add_channel("Chassis Yaw Rate", "deg/s", float, 2)
                self.channels["Chassis Yaw Rate"].messages = [Message(times[i], yaw_rate_val[i]) for i in range(len(times))]

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

            if "session.json" not in all_names:
                print("ERROR: Invalid RCZ file, missing session.json")
                return

            session_json = json.loads(z.read("session.json").decode("utf-8"))
            first_t = session_json.get("firstTimestamp", 0)

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
            for candidate in ["channel_1_200_0_1_1", "channel_1_200_0_2_1"]:
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
                "fastest_time": fastest_dur if fastest_dur != float("inf") else 0.0
            }
            
            # Helper to add channel messages
            def populate_channel(name, units, values_array, decimals=2):
                if len(values_array) != n_samples:
                    return
                filtered_vals = values_array[mask]
                self.add_channel(name, units, float, decimals)
                self.channels[name].decimals = decimals
                self.channels[name].messages = [Message(sample_times[i], filtered_vals[i]) for i in range(len(sample_times))]

            # Lap Number Channel
            populate_channel("Lap Number", "", lap_numbers, 0)
            # Running Time Channel
            populate_channel("Running Time", "s", times_sec, 2)

            # 2. Parse Speed
            if "channel_1_200_0_4_0" in namelist:
                raw_spd = np.frombuffer(read_channel("channel_1_200_0_4_0"), dtype="<i4")
                if len(raw_spd) == n_samples:
                    populate_channel("Ground Speed", "km/h", (raw_spd / 1000.0) * 3.6, 2)

            # 3. Parse Latitude & Longitude
            if "channel_1_200_0_3_1" in namelist:
                raw_ll = np.frombuffer(read_channel("channel_1_200_0_3_1"), dtype="<i4")
                if len(raw_ll) == n_samples * 2:
                    raw_ll = raw_ll.reshape(-1, 2)
                    populate_channel("GPS Latitude", "deg", raw_ll[:, 0] / 6000000.0, 7)
                    populate_channel("GPS Longitude", "deg", raw_ll[:, 1] / 6000000.0, 7)

            # 4. Parse Altitude
            if "channel_1_200_0_5_0" in namelist:
                raw_alt = np.frombuffer(read_channel("channel_1_200_0_5_0"), dtype="<i4")
                if len(raw_alt) == n_samples:
                    populate_channel("GPS Altitude", "m", raw_alt / 1000.0)

            # 5. Parse GPS Heading
            if "channel_1_200_0_6_0" in namelist:
                raw_hdg = np.frombuffer(read_channel("channel_1_200_0_6_0"), dtype="<i4")
                if len(raw_hdg) == n_samples:
                    populate_channel("GPS Heading", "deg", raw_hdg / 1000.0)

            # 6. Parse Accelerations
            if "channel_2_201_0_9_0" in namelist:
                raw_ay = np.frombuffer(read_channel("channel_2_201_0_9_0"), dtype="<i4")
                if len(raw_ay) == n_samples:
                    populate_channel("CG Accel Lateral", "G", raw_ay / 10000.0)

            if "channel_2_201_0_10_0" in namelist:
                raw_ax = np.frombuffer(read_channel("channel_2_201_0_10_0"), dtype="<i4")
                if len(raw_ax) == n_samples:
                    populate_channel("CG Accel Longitudinal", "G", raw_ax / 10000.0)

            if "channel_2_201_0_11_0" in namelist:
                raw_az = np.frombuffer(read_channel("channel_2_201_0_11_0"), dtype="<i4")
                if len(raw_az) == n_samples:
                    populate_channel("Lean Angle", "deg", raw_az / 10000.0)

            # 7. Parse Gyroscope / Yaw Rate
            if "channel_3_202_0_14_0" in namelist:
                raw_gz = np.frombuffer(read_channel("channel_3_202_0_14_0"), dtype="<i4")
                if len(raw_gz) == n_samples:
                    populate_channel("Chassis Yaw Rate", "deg/s", (raw_gz / 1000.0) * -1.0)
            elif "channel_3_202_0_12_0" in namelist:
                raw_gx = np.frombuffer(read_channel("channel_3_202_0_12_0"), dtype="<i4")
                if len(raw_gx) == n_samples:
                    populate_channel("x_rate_of_rotation", "", raw_gx / 1000.0)

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
                if not base_fname.startswith("channel_12_100_") or not base_fname.endswith("_1_1"):
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
                values = np.interp(times_sec, rel_times, raw_values)
                if pid in rcz_pid_map:
                    ch_name, ch_unit, ch_scale, ch_offset = rcz_pid_map[pid]
                    vals_processed = values * ch_scale + ch_offset
                    if pid == "1004":
                        vals_processed = np.round(vals_processed).clip(-1, 6)
                    populate_channel(ch_name, ch_unit, vals_processed)
                else:
                    populate_channel(f"OBD_{pid}", "", values)
        # Fallback for Chassis Yaw Rate if missing
        if "Chassis Yaw Rate" not in self.channels and "GPS Heading" in self.channels:
            gps_h_chan = self.channels["GPS Heading"]
            if len(gps_h_chan.messages) >= 2:
                times = np.array([m.timestamp for m in gps_h_chan.messages])
                headings = np.array([m.value for m in gps_h_chan.messages])
                h_unwrapped = np.unwrap(np.radians(headings))
                h_deg = np.degrees(h_unwrapped)
                yaw_rate_val = -np.gradient(h_deg, times)
                
                self.add_channel("Chassis Yaw Rate", "deg/s", float, 2)
                self.channels["Chassis Yaw Rate"].messages = [Message(times[i], yaw_rate_val[i]) for i in range(len(times))]

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
            print(channel_name)
            name, units = channel_name.split(" (")
            units = units[:-1]

            channel.name = name
            channel.units = units

    @staticmethod
    def __parse_can_log_line(line):
        """ Extracts the timestamp, bus, arbitration id, and data from a single line in a can log file
        recorded with candump -l.
        """
        stamp, bus, msg = line.split()
        stamp = float(stamp[1:-1])
        id, data = msg.split("#")
        id = int(id, 16)
        data = bytearray.fromhex(data)

        return stamp, bus, id, data

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
        if self.messages:
            return self.messages[0].timestamp
        else:
            return 0

    def end(self):
        if self.messages:
            return self.messages[-1].timestamp
        else:
            return 0

    def avg_frequency(self):
        """ Computes the average frequency from the samples based on the duration of the channel
        and the number of messages"""
        if len(self.messages) >= 2:
            dt = self.end() - self.start()
            return len(self.messages) / dt
        else:
            return 0

    def resample(self, start_time, end_time, frequency):
        """ Resamples the data such that all messages occur at a fixed frequency.

        If multiple messages fall within the time interval between messages for the new frequency,
        the latest message will be used. When no existing messages fall within the time interval
        the most recent value will be retained. If no existing message is present within the first
        new time interval, then the first message will be initialized at 0.
        """
        if not self.messages:
            return

        # Determine how many messages this channel should have,
        num_msgs = math.floor(frequency * (end_time - start_time))
        dt_step = 1.0 / frequency

        # Create a new message at each time new time point based on the frequency. As we step
        # through the new sample points we'll find the latest pre existing message to insert there,
        # and will hold that value until we find another message.
        value = 0
        t = start_time
        current_msgs_index = 0
        new_msgs = []
        for i in range(num_msgs):
            # Grab the latest message that falls in this time window, if there is one, and update
            # the current channel value
            while current_msgs_index < len(self.messages):
                msg_stamp = self.messages[current_msgs_index].timestamp

                if msg_stamp < t + 0.5 * dt_step:
                    # This message falls in the time window
                    value = self.messages[current_msgs_index].value
                    current_msgs_index += 1
                else:
                    # This messages belongs in a future window
                    break

            new_msgs.append(Message(t, value))
            t += dt_step

        self.messages = new_msgs

    def __str__(self):
        return "Channel: %s, Units: %s, Decimals: %d, Messages: %d, Frequency: %.2f Hz" % \
        (self.name, self.units, self.decimals, len(self.messages), self.avg_frequency())

class Message(object):
    """ A single message in a time series of data. """
    def __init__(self, timestamp=0, value=0):
        self.timestamp = float(timestamp)
        self.value = float(value)

    def __str__(self):
        return "t=%f, value=%f" % (self.timestamp, self.value)
