#!/usr/bin/env python3
"""
Aligns Assetto Corsa ACTI telemetry log files (Car Coord X, Car Coord Y) 
to Real-World WGS84 GPS coordinates for Thunderhill Raceway Park.

Mathematical Transformation:
  1. Translate ACTI coordinates relative to AC Start/Finish line (-72.557m, +166.100m)
  2. Rotate by 178.7 degrees CCW to correct North orientation (flipping AC +Y axis to True South 179.4 deg)
  3. Map to WGS84 GPS coordinates centered at Thunderhill S/F (39.540017 N, -122.331175 W)

Usage:
    python tools/align_acti_gps.py [path_to_acti_ld_or_dir] [--output_dir data/acti_aligned]
"""
import argparse
import os
import sys
import numpy as np
import json
from scipy.optimize import minimize
from scipy.spatial import cKDTree

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ldparser.ldparser import ldData
from data_log import DataLog, Message
from motec_log import MotecLog

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acti_track_gps.json")


def load_track_configs(config_path=None):
    cpath = config_path if config_path else CONFIG_FILE
    if os.path.isfile(cpath):
        try:
            with open(cpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load config {cpath}: {e}")
    return {}


def save_track_configs(configs, config_path=None):
    cpath = config_path if config_path else CONFIG_FILE
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)
    print(f"Saved track configs to '{cpath}'")


def transform_ac_to_gps(ac_x, ac_y, track_cfg):
    ref_lat = track_cfg.get("ref_lat", 39.540017)
    ref_lon = track_cfg.get("ref_lon", -122.331175)
    theta_deg = track_cfg.get("theta_deg", 180.019)
    dx_m = track_cfg.get("dx_m", -76.084)
    dy_m = track_cfg.get("dy_m", 58.019)

    theta_rad = np.radians(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    x_real_m = cos_t * ac_x - sin_t * ac_y + dx_m
    y_real_m = sin_t * ac_x + cos_t * ac_y + dy_m

    lat = ref_lat + (y_real_m / 111000.0)
    lon = ref_lon + (x_real_m / 85670.0)
    return lat, lon, x_real_m, y_real_m


def compute_gps_heading(x_real_m, y_real_m):
    """Computes course heading angle in degrees (0..360, 0=North, 90=East)."""
    dx = np.gradient(x_real_m)
    dy = np.gradient(y_real_m)
    dist_step = np.hypot(dx, dy)

    raw_heading = np.degrees(np.arctan2(dx, dy)) % 360.0

    moving = dist_step > 0.01  # > 1cm per sample step threshold
    if not np.any(moving):
        return np.zeros_like(raw_heading)

    hdg = np.copy(raw_heading)
    last_hdg = 0.0
    for i in range(len(hdg)):
        if moving[i]:
            last_hdg = hdg[i]
        else:
            hdg[i] = last_hdg

    first_mov = np.argmax(moving)
    if moving[first_mov]:
        hdg[:first_mov] = hdg[first_mov]

    return np.round(hdg, 3)


def auto_detect_track_key(input_ld, configs):
    lower_path = input_ld.lower()
    for key, cfg in configs.items():
        if key in lower_path or cfg.get("name", "").lower() in lower_path:
            return key
    for key in configs:
        if "thunderhill" in lower_path:
            return key
    return "thunderhill_east_bypass" if "thunderhill_east_bypass" in configs else list(configs.keys())[0]


def align_acti_file(input_ld, output_dir, track_cfg):
    try:
        ld = ldData.fromfile(input_ld)
    except Exception as e:
        print(f"ERROR reading {input_ld}: {e}")
        return False

    def get_ch(name):
        for c in ld.channs:
            if c.name == name:
                return c
        return None

    ch_x = get_ch("Car Coord X")
    ch_y = get_ch("Car Coord Y")
    ch_lat = get_ch("GPS Latitude")
    ch_lon = get_ch("GPS Longitude")

    if ch_x is not None and ch_y is not None:
        lat, lon, x_real_m, y_real_m = transform_ac_to_gps(ch_x.data, ch_y.data, track_cfg)
        freq = ch_x.freq if ch_x.freq > 0 else 20
    elif ch_lat is not None and ch_lon is not None:
        lat, lon = ch_lat.data, ch_lon.data
        ref_lat, ref_lon = float(np.mean(lat)), float(np.mean(lon))
        x_real_m = (lon - ref_lon) * 85670.0
        y_real_m = (lat - ref_lat) * 111000.0
        freq = ch_lat.freq if ch_lat.freq > 0 else 20
    else:
        print(f"SKIP {input_ld}: Missing Car Coord X/Y or GPS Latitude/Longitude channels")
        return False

    heading = compute_gps_heading(x_real_m, y_real_m)

    dl = DataLog()
    dl.metadata["venue_name"] = track_cfg.get("name", "Unknown Venue")
    dl.metadata["driver"] = getattr(ld.head, "driver", "")

    # Populate DataLog channels from ACTI log (filtering out Car Coord channels)
    for c in ld.channs:
        if c.name in ("Car Coord X", "Car Coord Y", "Car Coord Z"):
            continue
        dl.add_channel(c.name, c.unit, float, c.dec)
        freq = c.freq if c.freq > 0 else 20
        if len(c.data) > 0:
            times = np.linspace(0, len(c.data) / freq, len(c.data))
            dl.channels[c.name].messages = [Message(times[i], float(c.data[i])) for i in range(len(c.data))]

    # Add GPS Latitude / Longitude / Heading
    times = np.linspace(0, len(lat) / freq, len(lat))

    dl.add_channel("GPS Latitude", "deg", float, 7)
    dl.channels["GPS Latitude"].messages = [Message(times[i], float(lat[i])) for i in range(len(lat))]

    dl.add_channel("GPS Longitude", "deg", float, 7)
    dl.channels["GPS Longitude"].messages = [Message(times[i], float(lon[i])) for i in range(len(lon))]

    dl.add_channel("GPS Heading", "deg", float, 3)
    dl.channels["GPS Heading"].messages = [Message(times[i], float(heading[i])) for i in range(len(heading))]

    # Build MotecLog with full header metadata preserved
    ml = MotecLog()
    ml.driver = getattr(ld.head, "driver", "")
    ml.vehicle_id = getattr(ld.head, "vehicleid", "")
    ml.venue_name = track_cfg.get("name", "Unknown Venue")
    if hasattr(ld.head, "event") and ld.head.event:
        ml.event_name = getattr(ld.head.event, "name", "")
        ml.event_session = getattr(ld.head.event, "session", "")
    ml.short_comment = getattr(ld.head, "short_comment", "")
    ml.datetime = getattr(ld.head, "datetime", None)

    ml.initialize()
    ml.add_all_channels(dl)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_ld))[0]
    out_ld = os.path.join(output_dir, base_name + ".ld")
    out_ldx = os.path.join(output_dir, base_name + ".ldx")

    try:
        ml.write(out_ld)

        # Preserve original ACTI .ldx file containing all lap markers & sector info
        in_ldx = os.path.splitext(input_ld)[0] + ".ldx"
        if os.path.isfile(in_ldx):
            with open(in_ldx, "r", encoding="utf-8", errors="ignore") as f_in:
                content = f_in.read()
            with open(out_ldx, "w", encoding="utf-8") as f_out:
                f_out.write(content)
        print(f"  [OK] Aligned ({track_cfg.get('name', '')}) -> {out_ld}")
        return True
    except PermissionError:
        print(f"  [LOCKED/SKIP] File is currently open in MoTeC i2: {out_ld}")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed writing {out_ld}: {e}")
        return False


def calibrate_track(rw_ld_path, acti_ld_path, track_key, track_name=None):
    """ Auto-calibrates rotation and translation parameters using ICP optimization between real and sim logs. """
    print(f"Calibrating track '{track_key}' using ICP optimization...")
    rw_ld = ldData.fromfile(rw_ld_path)
    acti_ld = ldData.fromfile(acti_ld_path)

    def get_ch(ld, name):
        for c in ld.channs:
            if c.name == name:
                return c.data
        return None

    rw_lat = get_ch(rw_ld, "GPS Latitude")
    rw_lon = get_ch(rw_ld, "GPS Longitude")
    rw_spd = get_ch(rw_ld, "Ground Speed")
    mask_rw = rw_spd > 40.0

    ref_lat = float(rw_lat[mask_rw].mean())
    ref_lon = float(rw_lon[mask_rw].mean())

    rw_x = (rw_lon[mask_rw] - ref_lon) * 85670.0
    rw_y = (rw_lat[mask_rw] - ref_lat) * 111000.0
    rw_pts = np.column_stack([rw_x, rw_y])
    tree = cKDTree(rw_pts)

    ac_x = get_ch(acti_ld, "Car Coord X")
    ac_y = get_ch(acti_ld, "Car Coord Y")
    ac_spd = get_ch(acti_ld, "Ground Speed")
    mask_ac = ac_spd > 40.0
    ac_x_m = ac_x[mask_ac]
    ac_y_m = ac_y[mask_ac]

    def loss_func(params):
        theta, dx, dy = params
        tx = np.cos(theta) * ac_x_m - np.sin(theta) * ac_y_m + dx
        ty = np.sin(theta) * ac_x_m + np.cos(theta) * ac_y_m + dy
        dists, _ = tree.query(np.column_stack([tx, ty]))
        return np.mean(dists)

    res = minimize(loss_func, [np.radians(180.0), -50.0, 0.0], method="Powell")
    theta_deg = round(float(np.degrees(res.x[0]) % 360), 3)
    dx_m = round(float(res.x[1]), 3)
    dy_m = round(float(res.x[2]), 3)
    rmse = round(float(res.fun), 2)

    configs = load_track_configs()
    configs[track_key] = {
        "name": track_name if track_name else track_key.replace("_", " ").title(),
        "ref_lat": round(ref_lat, 6),
        "ref_lon": round(ref_lon, 6),
        "theta_deg": theta_deg,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "rmse_error_m": rmse
    }
    save_track_configs(configs)
    print(f"Calibration complete for '{track_key}': RMSE={rmse}m, theta={theta_deg} deg, dx={dx_m}m, dy={dy_m}m")
    return configs[track_key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Path to ACTI .ld file or folder")
    parser.add_argument("--track", help="Track config key (e.g. thunderhill_east_bypass, laguna_seca)")
    parser.add_argument("--config", help="Path to track config JSON file")
    parser.add_argument("--output_dir", default="data/acti_aligned", help="Output directory for aligned files")
    parser.add_argument("--calibrate", nargs=2, metavar=("REAL_LD", "ACTI_LD"),
                        help="Calibrate a new track using a real-world .ld and an ACTI .ld file")
    parser.add_argument("--track_key", help="Track key for calibration (e.g. laguna_seca)")
    parser.add_argument("--track_name", help="Human readable track name for calibration")
    args = parser.parse_args()

    configs = load_track_configs(args.config)

    if args.calibrate:
        rw_ld, acti_ld = args.calibrate
        t_key = args.track_key if args.track_key else "new_track"
        calibrate_track(rw_ld, acti_ld, t_key, args.track_name)
        sys.exit(0)

    if not args.input:
        parser.print_help()
        sys.exit(1)

    inp = os.path.expanduser(args.input)
    t_key = args.track if args.track else auto_detect_track_key(inp, configs)
    if t_key not in configs:
        print(f"ERROR: Track key '{t_key}' not found in configs: {list(configs.keys())}")
        sys.exit(1)

    track_cfg = configs[t_key]

    if os.path.isdir(inp):
        files = sorted([os.path.join(inp, f) for f in os.listdir(inp) if f.endswith(".ld")])
        print(f"Aligning {len(files)} ACTI files for track '{track_cfg.get('name', t_key)}' -> '{args.output_dir}'...\n")
        for f in files:
            align_acti_file(f, args.output_dir, track_cfg)
    elif os.path.isfile(inp):
        align_acti_file(inp, args.output_dir, track_cfg)
    else:
        print(f"ERROR: Input path '{inp}' does not exist.")
        sys.exit(1)


if __name__ == "__main__":
    main()
