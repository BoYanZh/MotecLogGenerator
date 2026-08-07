#!/usr/bin/env python3
"""
Tire grip comparison across all exported .ld sessions.

Metric: sustained lateral G (1s rolling floor) while Ground Speed > 60 km/h.
Columns:
  RawMax   - Peak instantaneous |ay| (includes bumps/spikes)
  SustMax  - Max 1s-sustained |ay| (floor of 1s sliding window, spike-free)
  P99      - 99th percentile of |ay| while moving
  BestSeg  - Duration (s) of the longest continuous >0.7G cornering segment
  SegMean  - Mean |ay| within that longest segment

Usage:
    python tools/analyze_tire_grip.py [--dir data/exported] [--min_spd 60] [--window 1.0]
"""
import argparse
import os
import sys

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ldparser.ldparser import ldData


def get_channel(ld, name):
    for c in ld.channs:
        if c.name == name:
            return c.data
    return None


def get_frequency(ld):
    if ld.channs:
        return ld.channs[0].freq
    return 20


def analyze_session(fpath, min_spd=60.0, window_sec=1.0):
    fname = os.path.basename(fpath)
    try:
        ld = ldData.fromfile(fpath)
    except Exception as e:
        print(f"  [ERROR] {fname}: {e}")
        return None

    ay = get_channel(ld, "CG Accel Lateral")
    spd = get_channel(ld, "Ground Speed")

    if ay is None:
        return None

    ay_abs = np.abs(ay)

    # Filter for moving vehicle (spd > min_spd)
    if spd is not None and len(spd) == len(ay):
        moving_mask = spd > min_spd
    else:
        moving_mask = np.ones(len(ay), dtype=bool)

    if not np.any(moving_mask):
        return None

    ay_mov = ay_abs[moving_mask]
    raw_max = float(np.max(ay_mov))
    p99 = float(np.percentile(ay_mov, 99))

    # Rolling window min to remove instantaneous spikes
    freq = get_frequency(ld)
    w_size = max(1, int(window_sec * freq))

    if len(ay_mov) >= w_size:
        padded = np.pad(ay_mov, (w_size // 2, w_size - 1 - w_size // 2), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, w_size)
        rolling_floor = np.min(windows, axis=1)[: len(ay_mov)]
        sust_max = float(np.max(rolling_floor))
    else:
        sust_max = raw_max

    # Find longest continuous cornering segment (> 0.7 G)
    thresh = 0.7
    in_corner = ay_mov > thresh
    best_len, best_start, curr_len = 0, 0, 0

    for i, active in enumerate(in_corner):
        if active:
            if curr_len == 0:
                start_idx = i
            curr_len += 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = start_idx
        else:
            curr_len = 0

    seg_duration = best_len / freq
    seg_mean = float(np.mean(ay_mov[best_start : best_start + best_len])) if best_len > 0 else 0.0

    return {
        "file": fname,
        "raw_max": raw_max,
        "sust_max": sust_max,
        "p99": p99,
        "seg_duration": seg_duration,
        "seg_mean": seg_mean,
        "laps": getattr(ld.head, "laps", 0),
        "vehicle": getattr(ld.head, "vehicleid", ""),
        "driver": getattr(ld.head, "driver", ""),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/exported", help="Directory containing .ld files")
    parser.add_argument("--min_spd", type=float, default=60.0, help="Min speed threshold in km/h (default: 60)")
    parser.add_argument("--window", type=float, default=1.0, help="Sustained G window in seconds (default: 1.0)")
    args = parser.parse_args()

    target_dir = os.path.expanduser(args.dir)
    if not os.path.isdir(target_dir):
        print(f"ERROR: Directory not found: '{target_dir}'")
        sys.exit(1)

    files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".ld")])
    if not files:
        print(f"No .ld files found in '{target_dir}'")
        sys.exit(0)

    print(f"\nAnalyzing {len(files)} MoTeC sessions in '{target_dir}' (min_speed={args.min_spd}km/h, window={args.window}s)...")
    print("=" * 95)
    print(f"{'Session File':<42} | {'RawMax':<7} | {'SustMax':<7} | {'P99':<6} | {'BestSeg':<8} | {'SegMean':<7}")
    print("-" * 95)

    results = []
    for f in files:
        res = analyze_session(f, min_spd=args.min_spd, window_sec=args.window)
        if res:
            results.append(res)
            print(
                f"{res['file']:<42} | {res['raw_max']:<7.2f} | {res['sust_max']:<7.2f} | {res['p99']:<6.2f} | {res['seg_duration']:<6.1f}s | {res['seg_mean']:<7.2f}"
            )

    print("=" * 95)
    if results:
        best_sust = max(results, key=lambda x: x["sust_max"])
        print(f"Highest Sustained Lateral Grip: {best_sust['file']} ({best_sust['sust_max']:.2f} G)")
    print()


if __name__ == "__main__":
    main()
