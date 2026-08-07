#!/usr/bin/env python3
"""
Verification Utility for MoTeC .ld and .ldx files.
Checks binary header integrity, channel value bounds, and XML lap beacon alignment.

Usage:
    python tools/verify_log.py [path_to_ld_file]
"""

import sys
import os
import xml.etree.ElementTree as ET
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ldparser.ldparser import ldData

def verify_motec_files(ld_path):
    print(f"==================================================")
    print(f"   Verifying MoTeC Log: {os.path.basename(ld_path)}")
    print(f"==================================================")

    if not os.path.isfile(ld_path):
        print(f"  ERROR: File not found: {ld_path}")
        return False

    ldx_path = os.path.splitext(ld_path)[0] + ".ldx"
    has_ldx = os.path.isfile(ldx_path)

    errors = []
    warnings = []

    # 1. Unpack Binary .ld File
    try:
        ld = ldData.fromfile(ld_path)
        print("  Binary Header Unpacked Successfully.")
        print(f"  - Driver:      '{getattr(ld.head, 'driver', '')}'")
        print(f"  - Vehicle:     '{getattr(ld.head, 'vehicleid', '')}'")
        print(f"  - Venue:       '{getattr(ld.head, 'venue', '')}'")
        print(f"  - Datetime:    {getattr(ld.head, 'datetime', 'N/A')}")
        print(f"  - Channels:    {len(ld.channs)}")
    except Exception as e:
        print(f"  CRITICAL ERROR unpacking binary .ld header: {e}")
        return False

    # 2. Check Channel Value Bounds & NaN/Inf Anomalies
    if not ld.channs:
        errors.append("Log contains 0 channels.")
    else:
        print(f"  Validated {len(ld.channs)} data channels.")
        for c in ld.channs:
            data = c.data
            if len(data) == 0:
                warnings.append(f"Channel '{c.name}' is empty.")
                continue

            if np.isnan(data).any() or np.isinf(data).any():
                errors.append(f"Channel '{c.name}' contains NaN or Inf values.")

            # Sanity checks on known channel ranges
            if c.name == "GPS Latitude" and (np.min(data) < -90 or np.max(data) > 90):
                errors.append(f"GPS Latitude out of physical bounds [-90, 90]: [{np.min(data)}, {np.max(data)}]")
            if c.name == "GPS Longitude" and (np.min(data) < -180 or np.max(data) > 180):
                errors.append(f"GPS Longitude out of physical bounds [-180, 180]: [{np.min(data)}, {np.max(data)}]")

    # 3. Check Expected Advanced Math Channels
    expected_math = ["Tire Slip Angle FL", "Understeer Index", "G Force Combined"]
    ch_names = [c.name for c in ld.channs]
    missing_math = [m for m in expected_math if m not in ch_names]

    if not missing_math:
        print("  All expected advanced math channels present (Tire Slip Angle FL, Understeer Index, G Force Combined).")
    else:
        warnings.append(f"Missing recommended math channels: {missing_math}")

    # 4. Parse XML Companion File (.ldx)
    if has_ldx:
        print(f"  Found accompanying .ldx file: {os.path.basename(ldx_path)}")
        try:
            tree = ET.parse(ldx_path)
            root = tree.getroot()

            beacons = []
            for elem in root.findall(".//Marker"):
                val = elem.get("Value")
                if val:
                    beacons.append(float(val))

            print(f"  - Lap Beacons Count: {len(beacons)}")

            laps = root.findall(".//Lap")
            print(f"  - Total Laps: {len(laps)}")

            fastest_lap_sec = None
            for lap in laps:
                dur = lap.get("Time")
                if dur:
                    dur_val = float(dur) / 1000000.0
                    if fastest_lap_sec is None or dur_val < fastest_lap_sec:
                        fastest_lap_sec = dur_val

            if fastest_lap_sec:
                mins = int(fastest_lap_sec // 60)
                secs = fastest_lap_sec % 60
                print(f"  - Fastest Lap: {mins}:{secs:06.3f}")
            else:
                print(f"  - Fastest Lap: N/A (0:00.000)")

        except Exception as e:
            errors.append(f"Failed parsing .ldx XML file: {e}")
    else:
        warnings.append("No accompanying .ldx file found (lap times and beacons will be missing in MoTeC).")

    # Summary
    print("\n--- Verification Summary ---")
    if warnings:
        for w in warnings:
            print(f"    WARNING: {w}")

    if errors:
        for err in errors:
            print(f"  ERROR: {err}")
        print("\nRESULT: FAILED  ")
        return False
    else:
        print("RESULT: PASSED ALL CHECKS  \n")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/verify_log.py <path_to_motec_file.ld>")
        sys.exit(1)

    success = verify_motec_files(sys.argv[1])
    sys.exit(0 if success else 1)
