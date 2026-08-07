"""
iRacing Mu Telemetry Exporter Converter & Cleaner (Legacy).

NOTE: For new iRacing telemetry, prefer the native .ibt parser in data_log.py.
Use `python motec_log_generator.py session.ibt AUTO` for direct .ibt-to-MoTeC
conversion without Mu Exporter. This tool remains for cleaning up existing
Mu Exporter-generated .ld files.

Converts 356-channel heavy iRacing Mu .ld files into standardized, lightweight
MoTeC .ld / .ldx files with:
  - Ground Speed auto-converted from m/s -> km/h (* 3.6).
  - True WGS84 GPS Latitude & GPS Longitude decimal degrees extraction.
  - Driver inputs mapped 100% strictly to canonical constants:
      - CH_BRAKE_POS ("Brake Pos")
      - CH_THROTTLE_POS ("Throttle Pos")
      - CH_STEERING_ANGLE ("Steering Angle")
      - CH_ENGINE_RPM ("Engine RPM")
      - CH_GEAR ("Gear")
      - CH_LAP_NUMBER ("Lap Number")
  - Advanced vehicle dynamics math channels (Slip Angle, Understeer Index, Yaw Rate, G Combined).
  - 94%+ file size reduction for fast MoTeC i2 loading.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldparser.ldparser import ldData
from data_log import DataLog, Message
from motec_log import MotecLog
from constants import (
    CH_GROUND_SPEED, CH_CG_ACCEL_LAT, CH_CG_ACCEL_LON, CH_GPS_LATITUDE, CH_GPS_LONGITUDE,
    CH_GPS_ALTITUDE, CH_LAP_NUMBER, CH_THROTTLE_POS, CH_BRAKE_POS, CH_ENGINE_RPM,
    CH_STEERING_ANGLE, CH_GEAR
)


def convert_iracing_mu_file(input_ld_path, output_ld_path):
    print(f"Loading iRacing Mu log: {input_ld_path}")
    ld = ldData.fromfile(input_ld_path)

    ch_dict = {ch.name: ch for ch in ld.channs}

    dlog = DataLog("iRacing Clean")
    dlog.datetime = ld.head.datetime

    first_ch = ld.channs[0]
    freq = float(first_ch.freq) if first_ch.freq > 0 else 60.0
    dt = 1.0 / freq
    n_samples = first_ch.data_len
    times = [i * dt for i in range(n_samples)]

    # 1. Ground Speed (m/s -> km/h) -> Strictly CH_GROUND_SPEED ("Ground Speed")
    speed_ch = ch_dict.get("Speed") or ch_dict.get("Ground Speed") or ch_dict.get("VelocityX")
    if speed_ch:
        dlog.add_channel(CH_GROUND_SPEED, "km/h", float, 2)
        raw_speed = speed_ch.data
        multiplier = 3.6 if speed_ch.unit.lower() == "m/s" or max(raw_speed) < 100 else 1.0
        for t, v in zip(times, raw_speed):
            dlog.channels[CH_GROUND_SPEED].messages.append(Message(t, float(v * multiplier)))

    # 2. GPS Latitude & Longitude extraction (WGS84 Decimal Degrees)
    combined_lat = None
    combined_lon = None

    if "Lat" in ch_dict or "GPS Latitude" in ch_dict:
        lat_ch = ch_dict.get("Lat") or ch_dict.get("GPS Latitude")
        combined_lat = list(lat_ch.data)
    elif "Latitude Degrees" in ch_dict:
        deg = ch_dict["Latitude Degrees"].data
        mins = ch_dict.get("Latitude Minutes", None)
        secs = ch_dict.get("Latitude Minute - fraction", None)
        mins_arr = mins.data if mins else [0] * n_samples
        secs_arr = secs.data if secs else [0] * n_samples
        combined_lat = [
            (abs(d) + abs(m) / 60.0 + abs(s) / 3600.0) if d >= 0 else -(abs(d) + abs(m) / 60.0 + abs(s) / 3600.0)
            for d, m, s in zip(deg, mins_arr, secs_arr)
        ]

    if "Lon" in ch_dict or "GPS Longitude" in ch_dict:
        lon_ch = ch_dict.get("Lon") or ch_dict.get("GPS Longitude")
        combined_lon = list(lon_ch.data)
    elif "Longitude Degrees" in ch_dict:
        deg = ch_dict["Longitude Degrees"].data
        mins = ch_dict.get("Longitude Minutes", None)
        secs = ch_dict.get("Longitude Minute - fraction", None)
        mins_arr = mins.data if mins else [0] * n_samples
        secs_arr = secs.data if secs else [0] * n_samples
        combined_lon = [
            (abs(d) + abs(m) / 60.0 + abs(s) / 3600.0) if d >= 0 else -(abs(d) + abs(m) / 60.0 + abs(s) / 3600.0)
            for d, m, s in zip(deg, mins_arr, secs_arr)
        ]

    if combined_lat:
        dlog.add_channel(CH_GPS_LATITUDE, "deg", float, 7)
        for t, v in zip(times, combined_lat):
            dlog.channels[CH_GPS_LATITUDE].messages.append(Message(t, float(v)))

    if combined_lon:
        dlog.add_channel(CH_GPS_LONGITUDE, "deg", float, 7)
        for t, v in zip(times, combined_lon):
            dlog.channels[CH_GPS_LONGITUDE].messages.append(Message(t, float(v)))

    # 3. GPS Altitude -> Strictly CH_GPS_ALTITUDE ("GPS Altitude")
    alt_ch = ch_dict.get("Alt") or ch_dict.get("GPS Altitude")
    if alt_ch:
        dlog.add_channel(CH_GPS_ALTITUDE, "m", float, 2)
        for t, v in zip(times, alt_ch.data):
            dlog.channels[CH_GPS_ALTITUDE].messages.append(Message(t, float(v)))

    # 4. Accelerations -> Strictly CH_CG_ACCEL_LAT ("CG Accel Lateral") & CH_CG_ACCEL_LON ("CG Accel Longitudinal")
    latacc_ch = ch_dict.get("LatAccel") or ch_dict.get("G Force Lat")
    if latacc_ch:
        dlog.add_channel(CH_CG_ACCEL_LAT, "G", float, 4)
        for t, v in zip(times, latacc_ch.data):
            dlog.channels[CH_CG_ACCEL_LAT].messages.append(Message(t, float(v)))

    longacc_ch = ch_dict.get("LongAccel") or ch_dict.get("G Force Long")
    if longacc_ch:
        dlog.add_channel(CH_CG_ACCEL_LON, "G", float, 4)
        for t, v in zip(times, longacc_ch.data):
            dlog.channels[CH_CG_ACCEL_LON].messages.append(Message(t, float(v)))

    # 5. Driver Controls -> Strictly CH_STEERING_ANGLE, CH_THROTTLE_POS, CH_BRAKE_POS
    steer_ch = ch_dict.get("SteeringWheelAngle") or ch_dict.get("Steering Angle")
    if steer_ch:
        dlog.add_channel(CH_STEERING_ANGLE, "deg", float, 2)
        mult = 57.2957795 if steer_ch.unit.lower() in ("rad", "") and max(abs(v) for v in steer_ch.data[:100]) < 10 else 1.0
        for t, v in zip(times, steer_ch.data):
            dlog.channels[CH_STEERING_ANGLE].messages.append(Message(t, float(v * mult)))

    throttle_ch = ch_dict.get("Throttle") or ch_dict.get("Throttle Position")
    if throttle_ch:
        dlog.add_channel(CH_THROTTLE_POS, "%", float, 2)
        mult = 100.0 if max(throttle_ch.data[:100]) <= 1.0 else 1.0
        for t, v in zip(times, throttle_ch.data):
            dlog.channels[CH_THROTTLE_POS].messages.append(Message(t, float(v * mult)))

    brake_ch = ch_dict.get("Brake") or ch_dict.get("Brake Pedal Position")
    if brake_ch:
        dlog.add_channel(CH_BRAKE_POS, "%", float, 2)
        mult = 100.0 if max(brake_ch.data[:100]) <= 1.0 else 1.0
        for t, v in zip(times, brake_ch.data):
            dlog.channels[CH_BRAKE_POS].messages.append(Message(t, float(v * mult)))

    rpm_ch = ch_dict.get("Engine RPM") or ch_dict.get("RPM")
    if rpm_ch:
        dlog.add_channel(CH_ENGINE_RPM, "rpm", float, 2)
        for t, v in zip(times, rpm_ch.data):
            dlog.channels[CH_ENGINE_RPM].messages.append(Message(t, float(v)))

    gear_ch = ch_dict.get("Gear")
    if gear_ch:
        dlog.add_channel(CH_GEAR, "", float, 0)
        for t, v in zip(times, gear_ch.data):
            dlog.channels[CH_GEAR].messages.append(Message(t, float(v)))

    # 6. Four Wheel Speeds (m/s -> km/h)
    wheel_map = {
        "Wheel Speed FL": ["Wheel Speed FL", "LFspeed"],
        "Wheel Speed FR": ["Wheel Speed FR", "RFspeed"],
        "Wheel Speed RL": ["Wheel Speed RL", "LRspeed"],
        "Wheel Speed RR": ["Wheel Speed RR", "RRspeed"],
    }
    for out_w_name, candidate_keys in wheel_map.items():
        wch = None
        for k in candidate_keys:
            if k in ch_dict:
                wch = ch_dict[k]
                break
        if wch:
            dlog.add_channel(out_w_name, "km/h", float, 2)
            mult = 3.6 if wch.unit.lower() == "m/s" or max(wch.data) < 100 else 1.0
            for t, v in zip(times, wch.data):
                dlog.channels[out_w_name].messages.append(Message(t, float(v * mult)))

    # 7. Auto-detect lap beacons from Lap channel transitions & add Lap Number / Corr Dist
    beacons = []
    lap_ch = ch_dict.get("Lap") or ch_dict.get("Lap Number")
    if lap_ch:
        lap_data = lap_ch.data
        dlog.add_channel(CH_LAP_NUMBER, "", float, 0)
        for t, v in zip(times, lap_data):
            dlog.channels[CH_LAP_NUMBER].messages.append(Message(t, float(v)))

        prev_l = lap_data[0]
        for i in range(1, len(lap_data)):
            curr_l = lap_data[i]
            if curr_l > prev_l and curr_l > 0:
                t = i * dt
                beacons.append((t, f"Lap {int(curr_l)}"))
                prev_l = curr_l

    dist_ch = ch_dict.get("Lap Distance")
    if dist_ch:
        dlog.add_channel("Corr Dist", "m", float, 2)
        for t, v in zip(times, dist_ch.data):
            dlog.channels["Corr Dist"].messages.append(Message(t, float(v)))

    print(f"Extracted {len(dlog.channels)} strict canonical channels.")

    # Calculate Advanced Vehicle Dynamics Math Channels
    dlog.calculate_math_channels()
    print(f"Calculated math channels (total {len(dlog.channels)} channels).")

    # Build MoTeC Log
    mlog = MotecLog()
    mlog.driver = ld.head.driver.strip() if ld.head.driver else "iRacing Driver"
    mlog.vehicle_id = ld.head.vehicleid.strip() if ld.head.vehicleid else "iRacing Vehicle"
    mlog.venue_name = ld.head.venue.strip() if ld.head.venue else "Laguna Seca"
    mlog.datetime = ld.head.datetime
    mlog.initialize()
    mlog.add_all_channels(dlog)

    os.makedirs(os.path.dirname(output_ld_path), exist_ok=True)
    mlog.write(output_ld_path)

    # Write .ldx file with auto-detected lap beacons and metadata
    output_ldx = os.path.splitext(output_ld_path)[0] + ".ldx"
    mlog.write_ldx(output_ldx, {}, beacons=beacons)
    print(f"Generated lap beacon index .ldx ({len(beacons)} beacons) -> {output_ldx}")

    print(f"DONE! Saved clean MoTeC log: {output_ld_path}")
    print(f"  Original size: {os.path.getsize(input_ld_path)/1024/1024:.2f} MB")
    print(f"  Clean size:    {os.path.getsize(output_ld_path)/1024/1024:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert iRacing Mu .ld file to standardized MoTeC .ld")
    parser.add_argument("input", help="Path to input iRacing Mu .ld file")
    parser.add_argument("--output", help="Path to output clean .ld file")
    args = parser.parse_args()

    out_p = args.output or os.path.splitext(args.input)[0] + "_clean.ld"
    convert_iracing_mu_file(args.input, out_p)
