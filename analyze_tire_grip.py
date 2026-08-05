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

# Known tire info per session date/name prefix
TIRE_INFO = {}


def get_channel(ld, name):
    for c in ld.channs:
        if c.name == name:
            return c.data
    return None


def analyze_session(fpath, min_spd=60.0, window_sec=1.0, freq=20):
    fname = os.path.basename(fpath)
    try:
        ld = ldData.fromfile(fpath)
    except Exception as e:
        return None, f"Read error: {e}"

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
    sustained = np.array([
        ay_abs[max(0, i - w): i + w].min()
        for i in range(len(ay_abs))
    ])

    sm = sustained[mask]
    am = ay_abs[mask]

    # Find longest segment above 0.7G
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
        segs.sort(key=lambda x: -(x[1] - x[0]))
        best_dur = (segs[0][1] - segs[0][0]) / freq
        best_mean = ay_abs[segs[0][0]: segs[0][1]].mean()

    # Determine tire info
    tire = "unknown"
    for key, val in TIRE_INFO.items():
        if key in fname:
            tire = val
            break

    return {
        "file": fname.replace(".ld", ""),
        "tire": tire,
        "raw_max": round(float(am.max()), 3),
        "sust_max": round(float(sm.max()), 3),
        "p99": round(float(np.percentile(am, 99)), 3),
        "p95": round(float(np.percentile(am, 95)), 3),
        "best_seg_s": round(best_dur, 1),
        "best_seg_mean": round(best_mean, 3),
        "moving_samples": int(mask.sum()),
    }, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/exported", help="Directory of .ld files")
    parser.add_argument("--min_spd", type=float, default=60.0, help="Min speed km/h to count as 'moving'")
    parser.add_argument("--window", type=float, default=1.0, help="Sustained G window size in seconds")
    args = parser.parse_args()

    dst = args.dir
    results = []
    skipped = []

    for f in sorted(os.listdir(dst)):
        if not f.endswith(".ld"):
            continue
        r, err = analyze_session(os.path.join(dst, f), args.min_spd, args.window)
        if r:
            results.append(r)
        else:
            skipped.append((f, err))

    # Sort by sustained max G descending
    results.sort(key=lambda x: -x["sust_max"])

    col_w = 58
    print()
    print("=" * 119)
    print(f"{'File':<{col_w}} {'Tire':<16} {'RawMax':>8} {'SustMax':>8} {'P99':>7} {'BestSeg':>8} {'SegMean':>8}")
    print("=" * 119)
    for r in results:
        short = r["file"]
        if len(short) > col_w - 1:
            short = short[-(col_w - 1):]
        print(
            f"{short:<{col_w}} {r['tire']:<16} "
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
