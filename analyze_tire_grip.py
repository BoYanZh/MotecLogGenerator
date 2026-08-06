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
    python analyze_tire_grip.py [--dir data/exported] [--min_spd 60] [--window 1.0]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
        return None, f"Read error: {e}"

    freq = get_frequency(ld)
    spd = get_channel(ld, "Ground Speed")
    ay = get_channel(ld, "CG Accel Lateral")

    if spd is None or ay is None:
        missing = []
        if spd is None:
            missing.append("Ground Speed")
        if ay is None:
            missing.append("CG Accel Lateral")
        return None, f"Missing: {', '.join(missing)}"

    mask = spd > min_spd
    if mask.sum() < 200:
        return None, f"Too few moving samples ({mask.sum()})"

    ay_abs = np.abs(ay)
    w = int(window_sec * freq)
    if w < 1:
        w = 1

    size = 2 * w + 1
    half = size // 2
    padded = np.pad(ay_abs, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    sustained = np.min(windows, axis=1)[:len(ay_abs)]

    sm = sustained[mask]
    am = ay_abs[mask]

    above = (ay_abs > 0.7).astype(int)
    segs, start = [], None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(above)))

    best_dur, best_mean = 0.0, 0.0
    if segs:
        best_seg = max(segs, key=lambda x: x[1] - x[0])
        best_dur = (best_seg[1] - best_seg[0]) / freq
        best_mean = ay_abs[best_seg[0]: best_seg[1]].mean()

    return {
        "file": fname.replace(".ld", ""),
        "raw_max": round(float(am.max()), 3),
        "sust_max": round(float(sm.max()), 3),
        "p99": round(float(np.percentile(am, 99, method="linear")), 3),
        "p95": round(float(np.percentile(am, 95, method="linear")), 3),
        "best_seg_s": round(best_dur, 1),
        "best_seg_mean": round(best_mean, 3),
        "moving_samples": int(mask.sum()),
    }, None


def load_tire_map(path="tire_map.json"):
    import json
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/exported", help="Directory of .ld files")
    parser.add_argument("--min_spd", type=float, default=60.0, help="Min speed km/h to count as 'moving'")
    parser.add_argument("--window", type=float, default=1.0, help="Sustained G window size in seconds")
    parser.add_argument("--tire_map", default="tire_map.json", help="Path to optional local tire_map.json")
    args = parser.parse_args()

    dst = args.dir
    if not os.path.isdir(dst):
        print(f"ERROR: Directory '{dst}' does not exist.")
        sys.exit(1)

    tire_map = load_tire_map(args.tire_map)
    results = []
    skipped = []

    for f in sorted(os.listdir(dst)):
        if not f.endswith(".ld"):
            continue
        r, err = analyze_session(os.path.join(dst, f), args.min_spd, args.window)
        if r:
            fname = r["file"]
            tire = tire_map.get(fname, "")
            if not tire:
                # Try matching key substrings
                for k, v in tire_map.items():
                    if k in fname:
                        tire = v
                        break
            r["tire"] = tire
            results.append(r)
        else:
            skipped.append((f, err))

    results.sort(key=lambda x: -x["sust_max"])

    has_tires = any(r.get("tire") for r in results)
    col_w = 48 if has_tires else 58

    print()
    print("=" * (112 if has_tires else 105))
    if has_tires:
        print(f"{'File':<{col_w}} {'Tire':<10} {'RawMax':>8} {'SustMax':>8} {'P99':>7} {'BestSeg':>8} {'SegMean':>8}")
    else:
        print(f"{'File':<{col_w}} {'RawMax':>8} {'SustMax':>8} {'P99':>7} {'BestSeg':>8} {'SegMean':>8}")
    print("=" * (112 if has_tires else 105))

    for r in results:
        short = r["file"]
        if len(short) > col_w - 1:
            short = short[-(col_w - 1):]
        if has_tires:
            t_str = r.get("tire", "")
            print(
                f"{short:<{col_w}} {t_str:<10} "
                f"{r['raw_max']:>7.3f}G {r['sust_max']:>7.3f}G "
                f"{r['p99']:>6.3f}G {r['best_seg_s']:>7.1f}s "
                f"{r['best_seg_mean']:>7.3f}G"
            )
        else:
            print(
                f"{short:<{col_w}} "
                f"{r['raw_max']:>7.3f}G {r['sust_max']:>7.3f}G "
                f"{r['p99']:>6.3f}G {r['best_seg_s']:>7.1f}s "
                f"{r['best_seg_mean']:>7.3f}G"
            )

    if skipped:
        print()
        print("Skipped (insufficient data):")
        for fname, reason in skipped:
            print(f"  {fname}: {reason}")

    print()
    print("Columns: RawMax=peak |ay|  SustMax=max 1s-sustained |ay|  P99=99th pctile")
    print("         BestSeg=longest >0.7G window  SegMean=avg G in that window")


if __name__ == "__main__":
    main()
