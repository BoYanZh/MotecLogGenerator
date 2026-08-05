#!/usr/bin/env python3
"""
Verification Utility for MoTeC .ld and .ldx files.
Checks binary header integrity, channel value bounds, and XML lap beacon alignment.
"""

import sys
import os
import xml.etree.ElementTree as ET
import numpy as np

# Ensure repository path is in sys.path
repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_dir)

from ldparser.ldparser import ldData

def verify_motec_files(ld_path):
    print(f"==================================================")
    print(f"   Verifying MoTeC Log: {os.path.basename(ld_path)}")
    print(f"==================================================")

    if not os.path.isfile(ld_path):
        print(f"❌ ERROR: File not found: {ld_path}")
        return False

    ldx_path = os.path.splitext(ld_path)[0] + ".ldx"
    has_ldx = os.path.isfile(ldx_path)

    errors = []
    warnings = []

    # 1. Binary Header Check
    try:
        ld = ldData.fromfile(ld_path)
        print(f"✓ Binary Header Unpacked Successfully.")
        print(f"  - Driver:      '{ld.head.driver}'")
        print(f"  - Vehicle:     '{ld.head.vehicleid}'")
        print(f"  - Venue:       '{ld.head.venue}'")
        print(f"  - Datetime:    {ld.head.datetime}")
        print(f"  - Channels:    {len(ld.channs)}")
    except Exception as e:
        print(f"❌ ERROR: Failed to parse binary header: {e}")
        return False

    # 2. Channel Data & Value Range Check
    if not ld.channs:
        errors.append("No data channels found in .ld file!")

    for idx, chan in enumerate(ld.channs):
        try:
            data = chan.data
            if len(data) != chan.data_len:
                errors.append(f"Channel '{chan.name}' length mismatch: expected {chan.data_len}, got {len(data)}")
            
            # Check for NaN / Inf
            if np.isnan(data).any() or np.isinf(data).any():
                errors.append(f"Channel '{chan.name}' contains NaN or Inf values!")

            min_val, max_val = data.min(), data.max()

            # Physical domain sanity checks
            if chan.name == "Ground Speed" and (min_val < -10.0 or max_val > 500.0):
                warnings.append(f"Ground Speed range suspicious: [{min_val:.2f}, {max_val:.2f}] km/h")
            elif "Latitude" in chan.name and (min_val < -90.0 or max_val > 90.0):
                errors.append(f"Latitude out of valid range [-90, 90]: [{min_val:.5f}, {max_val:.5f}]")
            elif "Longitude" in chan.name and (min_val < -180.0 or max_val > 180.0):
                errors.append(f"Longitude out of valid range [-180, 180]: [{min_val:.5f}, {max_val:.5f}]")

        except Exception as e:
            errors.append(f"Failed to read channel '{chan.name}': {e}")

    print(f"✓ Validated {len(ld.channs)} data channels.")

    # 3. Check Advanced Math Channels Presence
    expected_math_chans = ["Tire Slip Angle FL", "Understeer Index", "G Force Combined"]
    found_math = [c for c in expected_math_chans if c in ld]
    if len(found_math) == len(expected_math_chans):
        print(f"✓ All expected advanced math channels present ({', '.join(found_math)}).")
    else:
        warnings.append(f"Some math channels missing. Found: {found_math}")

    # 4. LDX File Verification
    if has_ldx:
        print(f"✓ Found accompanying .ldx file: {os.path.basename(ldx_path)}")
        try:
            tree = ET.parse(ldx_path)
            root = tree.getroot()

            markers = root.findall(".//Marker")
            print(f"  - Lap Beacons Count: {len(markers)}")
            prev_t = -1.0
            for m in markers:
                t_us = float(m.attrib.get("Time", 0))
                if t_us < prev_t:
                    errors.append(f"Non-monotonic beacon timestamp in .ldx: {t_us} < {prev_t}")
                prev_t = t_us

            details = {s.attrib["Id"]: s.attrib["Value"] for s in root.findall(".//Details/String")}
            print(f"  - Total Laps: {details.get('Total Laps', 'N/A')}")
            print(f"  - Fastest Lap: {details.get('Fastest Lap', 'N/A')} ({details.get('Fastest Time', 'N/A')})")

        except Exception as e:
            errors.append(f"Failed to parse XML in .ldx file: {e}")
    else:
        warnings.append("No accompanying .ldx file found.")

    # Summary
    print("\n--- Verification Summary ---")
    if warnings:
        for w in warnings:
            print(f"⚠️  WARNING: {w}")
    if errors:
        for err in errors:
            print(f"❌ ERROR: {err}")
        print("RESULT: FAILED ❌")
        return False
    else:
        print("RESULT: PASSED ALL CHECKS ✅")
        return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = r"data/racechrono/session_20260530_162048_thunder_hill_ccw_lap7_v3.ld"

    verify_motec_files(target)
