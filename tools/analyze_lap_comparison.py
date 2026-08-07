#!/usr/bin/env python3
"""
Lap Time & Sector Delta Comparison Utility for MoTeC .ld and .ldx files.

Inspects exported MoTeC session logs, extracts lap beacon timestamps, lap times,
fastest lap benchmarks, and sector splits across sessions.

Usage:
    python tools/compare_laps.py [--dir data/exported] [--fastest_only]
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ldparser.ldparser import ldData


def parse_time_str(time_str):
    """ Converts '2:07.710' or '127.710' into seconds (float). """
    if not time_str:
        return None
    try:
        if ":" in time_str:
            parts = time_str.split(":")
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(time_str)
    except Exception:
        return None


def parse_ldx_laps(ldx_path):
    """ Parses lap times and beacon timestamps from a MoTeC companion .ldx file. """
    if not os.path.isfile(ldx_path):
        return [], None, None

    try:
        tree = ET.parse(ldx_path)
        root = tree.getroot()

        # Extract Details metadata if present
        fastest_str = None
        total_laps_str = None
        for string_elem in root.findall(".//String"):
            sid = string_elem.get("Id")
            if sid == "Fastest Time":
                fastest_str = string_elem.get("Value")
            elif sid == "Total Laps":
                total_laps_str = string_elem.get("Value")

        fastest_sec_meta = parse_time_str(fastest_str) if fastest_str else None

        # 1. Try parsing explicit Lap elements (<Lap Time="..." />)
        laps_data = []
        for idx, lap_elem in enumerate(root.findall(".//Lap")):
            dur_raw = lap_elem.get("Time")
            if not dur_raw:
                continue
            dur_sec = float(dur_raw) / 1000000.0

            splits_raw = lap_elem.get("Split")
            splits_sec = []
            if splits_raw:
                for s in splits_raw.split(";"):
                    if s.strip():
                        splits_sec.append(float(s) / 1000000.0)

            laps_data.append({
                "lap_num": idx + 1,
                "duration_sec": dur_sec,
                "splits": splits_sec
            })

        if laps_data:
            return laps_data, fastest_sec_meta, len(laps_data)

        # 2. Parse lap times from Beacon Markers (<Marker Name="..." Time="..." />)
        beacons = []
        for marker in root.findall(".//MarkerGroup[@Name='Beacons']//Marker"):
            t_raw = marker.get("Time")
            if t_raw:
                try:
                    beacons.append(float(t_raw) / 1000000.0)
                except ValueError:
                    pass

        if not beacons:
            # Try any Marker elements
            for marker in root.findall(".//Marker"):
                t_raw = marker.get("Time")
                if t_raw:
                    try:
                        beacons.append(float(t_raw) / 1000000.0)
                    except ValueError:
                        pass

        beacons.sort()

        if len(beacons) >= 2:
            for i in range(len(beacons) - 1):
                dur = beacons[i + 1] - beacons[i]
                if dur > 15.0:  # Filter out false beacons / short intervals
                    laps_data.append({
                        "lap_num": len(laps_data) + 1,
                        "duration_sec": dur,
                        "splits": []
                    })

        return laps_data, fastest_sec_meta, len(laps_data)
    except Exception as e:
        print(f"WARNING: Failed parsing .ldx {ldx_path}: {e}")
        return [], None, None


def format_lap_time(sec):
    if sec is None or sec <= 0:
        return "N/A"
    mins = int(sec // 60)
    rem_sec = sec % 60
    return f"{mins}:{rem_sec:06.3f}"


def analyze_session_laps(ld_path):
    fname = os.path.basename(ld_path)
    ldx_path = os.path.splitext(ld_path)[0] + ".ldx"

    try:
        ld = ldData.fromfile(ld_path)
    except Exception as e:
        return None

    driver = getattr(ld.head, "driver", "Unknown") or "Unknown"
    vehicle = getattr(ld.head, "vehicleid", "") or ""
    venue = getattr(ld.head, "venue", "") or ""
    dt = getattr(ld.head, "datetime", None)
    dt_str = dt.strftime("%Y-%m-%d %H:%M") if dt else "N/A"

    laps, fastest_meta, total_laps_count = parse_ldx_laps(ldx_path)

    valid_durations = [l["duration_sec"] for l in laps if l["duration_sec"] > 15.0]
    fastest_sec = min(valid_durations) if valid_durations else fastest_meta

    return {
        "file": fname,
        "driver": driver,
        "vehicle": vehicle,
        "venue": venue,
        "datetime": dt_str,
        "total_laps": len(laps) if laps else total_laps_count or 0,
        "fastest_sec": fastest_sec,
        "laps": laps
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/exported", help="Directory containing .ld files (default: data/exported)")
    parser.add_argument("--fastest_only", action="store_true", help="Only show the fastest lap per session")
    args = parser.parse_args()

    target_dir = os.path.expanduser(args.dir)
    if not os.path.isdir(target_dir):
        print(f"ERROR: Directory not found: '{target_dir}'")
        sys.exit(1)

    files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".ld")])
    if not files:
        print(f"No .ld files found in '{target_dir}'")
        sys.exit(0)

    results = []
    for f in files:
        res = analyze_session_laps(f)
        if res:
            results.append(res)

    if not results:
        print("No valid sessions analyzed.")
        sys.exit(0)

    valid_sessions = [r for r in results if r["fastest_sec"] is not None]
    valid_sessions.sort(key=lambda x: x["fastest_sec"])

    overall_best_sec = valid_sessions[0]["fastest_sec"] if valid_sessions else None

    print(f"\n==========================================================================================================")
    print(f"   LAP TIME & TELEMETRY LEADERBOARD (Analyzed {len(files)} sessions in '{target_dir}')")
    print(f"==========================================================================================================")
    header = f"{'Rank':<5} | {'Fastest Lap':<11} | {'Gap (s)':<8} | {'Driver':<15} | {'Venue / Track':<32} | {'Date':<10}"
    print(header)
    print("-" * 106)

    for rank, s in enumerate(valid_sessions, 1):
        fastest_fmt = format_lap_time(s["fastest_sec"])
        gap = f"+{s['fastest_sec'] - overall_best_sec:.3f}s" if rank > 1 else "BEST"
        venue_short = s['venue'][:32] if s['venue'] else "N/A"
        date_short = s['datetime'][:10] if s['datetime'] else "N/A"
        print(f"{rank:<5} | {fastest_fmt:<11} | {gap:<8} | {s['driver']:<15} | {venue_short:<32} | {date_short:<10}")

    print("=" * 106)

    if not args.fastest_only:
        print("\n--- DETAILED SESSION BREAKDOWN ---")
        for s in valid_sessions:
            print(f"\n  Session: {s['file']}")
            print(f"   Driver: {s['driver']} | Vehicle: {s['vehicle']} | Venue: {s['venue']} | Date: {s['datetime']}")
            print(f"   Total Laps: {s['total_laps']} | Fastest Lap: {format_lap_time(s['fastest_sec'])}")
            if s["laps"]:
                print("   Laps Detail:")
                for l in s["laps"]:
                    is_fastest = "   (FASTEST)" if s["fastest_sec"] and abs(l["duration_sec"] - s["fastest_sec"]) < 0.001 else ""
                    splits_fmt = " | ".join([f"S{i+1}: {format_lap_time(sp)}" for i, sp in enumerate(l["splits"])])
                    splits_str = f" [{splits_fmt}]" if l["splits"] else ""
                    print(f"     - Lap {l['lap_num']:<2}: {format_lap_time(l['duration_sec'])}{splits_str}{is_fastest}")


if __name__ == "__main__":
    main()
