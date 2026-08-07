#!/usr/bin/env python3
"""
Corner-by-Corner Time Loss Analysis Utility for MoTeC .ld Log Files.

Calculates spatial distance-based time delta trace dt(s) between a Target Lap
and a Reference (Benchmark) Lap, analyzing corner-by-corner time loss, apex speed
differences (V_min), and primary causes (early braking, low apex speed, late throttle).

Usage:
    python tools/analyze_corner_time_loss.py <ref_log.ld> <target_log.ld>
    python tools/analyze_corner_time_loss.py --dir data/exported
"""

import argparse
import os
import sys
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ldparser.ldparser import ldData

# Pre-defined corner distance segments for Thunderhill East Bypass (meters)
THUNDERHILL_CORNER_SEGMENTS = [
    {"name": "Turn 1 (Fast Left)",       "start_m": 122.0,  "end_m": 450.0},
    {"name": "Turn 2 (Long Sweeper)",    "start_m": 450.0,  "end_m": 858.0},
    {"name": "Turn 3 (Off-Camber)",      "start_m": 858.0,  "end_m": 1250.0},
    {"name": "Turn 4/5 (Cyclone Hill)",  "start_m": 1250.0, "end_m": 1650.0},
    {"name": "Turn 6 (S-Turn Exit)",     "start_m": 1650.0, "end_m": 2000.0},
    {"name": "Turn 8 (Bypass Sweeper)",  "start_m": 2000.0, "end_m": 2444.0},
    {"name": "Turn 9 (Uphill Crest)",    "start_m": 2444.0, "end_m": 2800.0},
    {"name": "Turn 14/15 (Final Complex)","start_m": 2800.0, "end_m": 3100.0},
]


def extract_best_lap_spatial_data(ld_path, resample_step_m=1.0):
    """ Extracts and interpolates Ground Speed v(s) onto a uniform 1m distance grid. """
    try:
        ld = ldData.fromfile(ld_path)
    except Exception as e:
        print(f"ERROR reading {ld_path}: {e}")
        return None

    def get_ch(name):
        for c in ld.channs:
            if c.name == name:
                return c.data
        return None

    spd_kmh = get_ch("Ground Speed")
    if spd_kmh is None or len(spd_kmh) < 100:
        return None

    freq = ld.channs[0].freq if ld.channs else 20
    dt_sample = 1.0 / freq
    spd_ms = np.maximum(spd_kmh / 3.6, 0.5)

    # Compute cumulative spatial distance s(t)
    dist = np.cumsum(spd_ms * dt_sample)
    total_dist_m = dist[-1]

    # Uniform distance grid (1m resolution)
    s_grid = np.arange(0.0, total_dist_m, resample_step_m)

    # Interpolate speed v(s) on spatial grid
    v_ms_grid = np.interp(s_grid, dist, spd_ms)
    v_kmh_grid = v_ms_grid * 3.6

    # Calculate cumulative time t(s) = sum(ds / v(s))
    dt_grid = resample_step_m / v_ms_grid
    t_grid = np.cumsum(dt_grid)

    return {
        "file": os.path.basename(ld_path),
        "driver": getattr(ld.head, "driver", "Unknown") or "Unknown",
        "vehicle": getattr(ld.head, "vehicleid", "") or "",
        "venue": getattr(ld.head, "venue", "") or "",
        "s_grid": s_grid,
        "v_kmh": v_kmh_grid,
        "t_grid": t_grid,
        "total_dist_m": total_dist_m,
        "lap_time_sec": t_grid[-1]
    }


def analyze_corner_deltas(ref_lap, target_lap, corners=THUNDERHILL_CORNER_SEGMENTS):
    """ Computes spatial time deltas and corner-by-corner time loss. """
    max_dist = min(ref_lap["total_dist_m"], target_lap["total_dist_m"])
    s_grid = np.arange(0.0, max_dist, 1.0)

    # Interpolate speed and time onto common distance grid
    ref_v = np.interp(s_grid, ref_lap["s_grid"], ref_lap["v_kmh"])
    target_v = np.interp(s_grid, target_lap["s_grid"], target_lap["v_kmh"])

    ref_t = np.interp(s_grid, ref_lap["s_grid"], ref_lap["t_grid"])
    target_t = np.interp(s_grid, target_lap["s_grid"], target_lap["t_grid"])

    # Calculate continuous time delta trace dt(s) = t_target(s) - t_ref(s)
    # Ref lap is benchmark; dt > 0 means target lost time.
    dt_trace = target_t - ref_t

    corner_results = []

    for c in corners:
        start_m = c["start_m"]
        end_m = c["end_m"]

        if start_m >= max_dist:
            continue
        end_m = min(end_m, max_dist)

        idx_range = np.where((s_grid >= start_m) & (s_grid <= end_m))[0]
        if len(idx_range) < 5:
            continue

        # Corner Entry & Exit time deltas
        dt_entry = dt_trace[idx_range[0]]
        dt_exit = dt_trace[idx_range[-1]]
        time_loss_sec = dt_exit - dt_entry

        # Apex speed (minimum speed in corner segment)
        ref_vmin = float(np.min(ref_v[idx_range]))
        target_vmin = float(np.min(target_v[idx_range]))
        vmin_diff_kmh = target_vmin - ref_vmin

        # Time spent inside corner segment
        ref_time = ref_t[idx_range[-1]] - ref_t[idx_range[0]]
        target_time = target_t[idx_range[-1]] - target_t[idx_range[0]]

        # Primary Cause Diagnostic
        if time_loss_sec > 0.05:
            if vmin_diff_kmh < -3.0:
                cause = "Low Apex Speed / Over-braking"
            elif ref_v[idx_range[0]] - target_v[idx_range[0]] > 4.0:
                cause = "Early Braking on Entry"
            elif ref_v[idx_range[-1]] - target_v[idx_range[-1]] > 4.0:
                cause = "Hesitant / Late Throttle Application"
            else:
                cause = "General Cornering Time Loss"
        elif time_loss_sec < -0.05:
            cause = "Gained Time (Faster Cornering)"
        else:
            cause = "Equal Pace"

        corner_results.append({
            "name": c["name"],
            "start_m": start_m,
            "end_m": end_m,
            "time_loss_sec": time_loss_sec,
            "ref_vmin": ref_vmin,
            "target_vmin": target_vmin,
            "vmin_diff": vmin_diff_kmh,
            "ref_time": ref_time,
            "target_time": target_time,
            "cause": cause
        })

    return corner_results, dt_trace[-1]


def print_corner_loss_report(ref_lap, target_lap, corner_results, total_loss_sec):
    print("\n" + "=" * 112)
    print(f"   CORNER-BY-CORNER TIME LOSS & DELTA ANALYSIS")
    print(f"   Reference Lap (Benchmark): {ref_lap['file']} ({ref_lap['driver']}) - Lap Time: {ref_lap['lap_time_sec']:.3f}s")
    print(f"   Target Lap (Comparison):   {target_lap['file']} ({target_lap['driver']}) - Lap Time: {target_lap['lap_time_sec']:.3f}s")
    print(f"   Total Time Difference:     {total_loss_sec:+.3f}s ({'SLOWER' if total_loss_sec > 0 else 'FASTER'})")
    print("=" * 112)
    header = f"{'Corner Segment':<26} | {'Time Loss':<10} | {'Ref Vmin':<9} | {'Tgt Vmin':<9} | {'Δ Vmin':<8} | {'Primary Cause / Diagnostic':<30}"
    print(header)
    print("-" * 112)

    total_corner_loss = 0.0

    for c in corner_results:
        loss_str = f"{c['time_loss_sec']:+.3f}s"
        vmin_diff_str = f"{c['vmin_diff']:+.1f} km/h"

        if c['time_loss_sec'] > 0.05:
            total_corner_loss += c['time_loss_sec']

        print(f"{c['name']:<26} | {loss_str:<10} | {c['ref_vmin']:<6.1f} km/h | {c['target_vmin']:<6.1f} km/h | {vmin_diff_str:<8} | {c['cause']:<30}")

    print("=" * 112)
    print(f"Summary: Total Cumulative Time Lost in Corners: {total_corner_loss:+.3f}s")
    print("=" * 112 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref_log", nargs="?", help="Path to reference (benchmark) .ld log file")
    parser.add_argument("target_log", nargs="?", help="Path to target (comparison) .ld log file")
    parser.add_argument("--dir", help="Directory containing .ld files (auto-selects fastest as reference)")
    args = parser.parse_args()

    if args.dir:
        target_dir = os.path.expanduser(args.dir)
        files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".ld")])
        if len(files) < 2:
            print("ERROR: Need at least 2 .ld files in directory for comparison.")
            sys.exit(1)

        laps_data = []
        for f in files:
            d = extract_best_lap_spatial_data(f)
            if d:
                laps_data.append(d)

        if len(laps_data) < 2:
            print("ERROR: Could not extract spatial data from files.")
            sys.exit(1)

        laps_data.sort(key=lambda x: x["lap_time_sec"])
        ref_lap = laps_data[0]

        for target_lap in laps_data[1:]:
            c_results, tot_loss = analyze_corner_deltas(ref_lap, target_lap)
            print_corner_loss_report(ref_lap, target_lap, c_results, tot_loss)
        sys.exit(0)

    if not args.ref_log or not args.target_log:
        parser.print_help()
        sys.exit(1)

    ref_lap = extract_best_lap_spatial_data(args.ref_log)
    target_lap = extract_best_lap_spatial_data(args.target_log)

    if not ref_lap or not target_lap:
        print("ERROR: Could not extract spatial data from input files.")
        sys.exit(1)

    c_results, tot_loss = analyze_corner_deltas(ref_lap, target_lap)
    print_corner_loss_report(ref_lap, target_lap, c_results, tot_loss)


if __name__ == "__main__":
    main()
