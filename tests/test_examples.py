"""Smoke and regression tests using the example/ input files."""

import os
import sys
import csv
import importlib
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from unittest import SkipTest
from unittest.mock import patch

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from data_log import DataLog, Message

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _read_lines(filename):
    path = os.path.join(EXAMPLES, filename)
    with open(path) as f:
        return f.readlines()


def _dedup_channels(log):
    """MoTeC doesn't allow duplicate channel names, validate this invariant."""
    assert len(log.channels) == len(set(log.channels)), f"Duplicate channel names: {list(log.channels)}"


def _import_or_skip(module_name):
    """Import an optional test dependency without turning a skip into a pass."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SkipTest(f"{module_name} not installed") from exc


def _sha256(path):
    with open(path, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


# ---------------------------------------------------------------------------
# CSV & PB BUDDY
# ---------------------------------------------------------------------------
def test_csv_smoke():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))


def test_pbbuddy_smoke():
    log = DataLog()
    lines = _read_lines("pbbuddy_sample.csv")
    log.from_pbbuddy_log(lines)
    _dedup_channels(log)
    assert len(log.channels) > 0
    assert "Ground Speed" in log.channels
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.metadata.get("venue_name") == "Thunderhill East (Cyclone)"
    assert log.duration() > 0


def test_vbo_smoke():
    log = DataLog()
    lines = _read_lines("vbo_sample.vbo")
    log.from_vbo_log(lines)
    _dedup_channels(log)
    assert len(log.channels) > 0
    assert "Ground Speed" in log.channels
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert "CG Accel Lateral" in log.channels
    assert "CG Accel Longitudinal" in log.channels
    assert log.datetime.strftime("%Y-%m-%d") == "2026-01-01"
    assert log.duration() > 0


def test_ibt_smoke():
    log = DataLog()
    log.from_ibt_log(os.path.join(EXAMPLES, "ibt_sample.ibt"))

    assert "Ground Speed" in log.channels
    assert "Lap Number" in log.channels
    assert len(log.channels["Ground Speed"].messages) == 10

    spd_msgs = log.channels["Ground Speed"].messages
    assert abs(spd_msgs[-1].value - 45.0 * 3.6) < 0.1

    lap_msgs = log.channels["Lap Number"].messages
    assert lap_msgs[-1].value == 1.0

    assert log.metadata["venue_name"] == "Test Track"
    assert log.metadata["driver"] == "Test Driver"
    assert log.metadata["vehicle_id"] == "Test Car"
    assert log.datetime is not None

    beacons = log.laps_info.get("beacons", [])
    assert len(beacons) == 1
    assert "Lap 1" in beacons[0][1]

    _dedup_channels(log)


def test_ibt_invalid():
    import tempfile
    import os as _os

    with tempfile.NamedTemporaryFile(suffix=".ibt", delete=False) as tf:
        tf.write(b"\x00" * 10)
        tf.flush()
        tmp_path = tf.name

    try:
        log = DataLog()
        log.from_ibt_log(tmp_path)
        assert len(log.channels) == 0
    finally:
        _os.unlink(tmp_path)

    import struct
    header = struct.pack("12i", 2, 0, 10, 0, 0, 0, 0, 0, 0, 8, 0, 0)
    buf_info = struct.pack("4i", 0, 999999, 0, 0)
    buf = header + buf_info + b"\x00" * 200

    with tempfile.NamedTemporaryFile(suffix=".ibt", delete=False) as tf:
        tf.write(buf)
        tf.flush()
        tmp_path2 = tf.name

    try:
        log = DataLog()
        log.from_ibt_log(tmp_path2)
        assert len(log.channels) == 0
    finally:
        _os.unlink(tmp_path2)


def test_aim_solo_smoke():
    log = DataLog()
    lines = _read_lines("aim_solo_sample.csv")
    log.from_racechrono_log(lines)
    _dedup_channels(log)
    assert len(log.channels) > 0
    assert "Ground Speed" in log.channels
    assert "CG Accel Lateral" in log.channels
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.metadata.get("venue_name") == "Laguna Seca"
    assert log.datetime.strftime("%Y-%m-%d") == "2026-02-14"
    assert log.duration() > 0


def test_g_source_modes():
    # 1. Test 'calc' mode on AiM log: forces deriving G from GPS
    log1 = DataLog()
    log1.from_racechrono_log(_read_lines("aim_solo_sample.csv"))
    log1.calculate_math_channels(g_source="calc")
    assert "CG Accel Lateral" in log1.channels
    assert "CG Accel Longitudinal" in log1.channels

    # 2. Test 'sensor' mode on PB Buddy log (no raw sensor Gs): leaves G channels derived-free
    log2 = DataLog()
    log2.from_pbbuddy_log(_read_lines("pbbuddy_sample.csv"))
    log2.calculate_math_channels(g_source="sensor")
    assert "CG Accel Lateral" not in log2.channels

    # 3. Test 'auto' mode on PB Buddy log: automatically derives Gs from GPS
    log3 = DataLog()
    log3.from_pbbuddy_log(_read_lines("pbbuddy_sample.csv"))
    log3.calculate_math_channels(g_source="auto")
    assert "CG Accel Lateral" in log3.channels


def test_csv_resample_and_math():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))
    log.resample(20)
    assert log.duration() > 0
    for ch in log.channels.values():
        assert len(ch.messages) > 0, f"Channel {ch.name} empty after resample"
    log.calculate_math_channels()


# ---------------------------------------------------------------------------
# Accessport
# ---------------------------------------------------------------------------
def test_accessport_smoke():
    log = DataLog()
    log.from_accessport_log(_read_lines("accessport_sample.csv"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert log.duration() > 0
    _dedup_channels(log)


def test_accessport_resample_and_math():
    log = DataLog()
    log.from_accessport_log(_read_lines("accessport_sample.csv"))
    log.resample(20)
    assert log.duration() > 0
    log.calculate_math_channels()


# ---------------------------------------------------------------------------
# RaceChrono
# ---------------------------------------------------------------------------
def test_racechrono_smoke():
    log = DataLog()
    log.from_racechrono_log(_read_lines("racechrono_sample.csv"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert log.duration() > 0
    _dedup_channels(log)


def test_racechrono_resample_and_math():
    log = DataLog()
    log.from_racechrono_log(_read_lines("racechrono_sample.csv"))
    log.resample(20)
    assert log.duration() > 0
    log.calculate_math_channels()


# ---------------------------------------------------------------------------
# CAN
# ---------------------------------------------------------------------------
def test_can_smoke():
    cantools = _import_or_skip("cantools")

    log = DataLog()
    dbc_path = os.path.join(EXAMPLES, "sample_can_spec.dbc")
    can_db = cantools.database.load_file(dbc_path)
    log.from_can_log(_read_lines("can_sample.log"), can_db)
    assert len(log.channels) >= 1, f"Too few channels: {len(log.channels)}"
    assert log.duration() > 0
    _dedup_channels(log)


def test_can_resample():
    cantools = _import_or_skip("cantools")

    log = DataLog()
    dbc_path = os.path.join(EXAMPLES, "sample_can_spec.dbc")
    can_db = cantools.database.load_file(dbc_path)
    log.from_can_log(_read_lines("can_sample.log"), can_db)
    log.resample(20)
    assert log.duration() > 0


# ---------------------------------------------------------------------------
# Discrete channel resample behaviour
# ---------------------------------------------------------------------------
def test_discrete_channels_use_zoh():
    log = DataLog()
    log.add_channel("Gear", "", float, 0)
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Gear"].messages = [
        type("M", (), {"timestamp": 0.0, "value": 2.0})(),
        type("M", (), {"timestamp": 1.0, "value": 3.0})(),
    ]
    log.channels["Ground Speed"].messages = [
        type("M", (), {"timestamp": 0.0, "value": 0.0})(),
        type("M", (), {"timestamp": 1.0, "value": 100.0})(),
    ]
    log.resample(10)

    gear_vals = [int(m.value) for m in log.channels["Gear"].messages]
    spd_vals = [m.value for m in log.channels["Ground Speed"].messages]

    # Gear must stay integer (ZOH)
    assert all(v in (2, 3) for v in gear_vals), f"Gear values not discrete: {gear_vals}"
    # Speed gets linear interp
    assert any(0 < v < 100 for v in spd_vals), f"Speed not interpolated: {spd_vals}"


def test_mask_interp_gaps_default():
    log = DataLog()
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].messages = [
        Message(0.0, 10.0),
        Message(3.0, 40.0),
    ]
    log.resample(10, mask_interp_gaps=False)
    vals = [m.value for m in log.channels["Ground Speed"].messages]
    assert not any(np.isnan(v) for v in vals)
    mid_idx = len(vals) // 2
    assert 10.0 < vals[mid_idx] < 40.0


def test_mask_interp_gaps_enabled():
    log = DataLog()
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].messages = [
        Message(0.0, 10.0),
        Message(3.0, 40.0),
    ]
    log.resample(10, mask_interp_gaps=True)
    vals = [m.value for m in log.channels["Ground Speed"].messages]
    mid_idx = len(vals) // 2
    assert np.isnan(vals[mid_idx])


def test_compute_gps_heading():
    from tools.convert_acti import compute_gps_heading
    # Moving North: x=0, y increases -> 0 deg
    x = np.zeros(10)
    y = np.linspace(0, 100, 10)
    hdg_north = compute_gps_heading(x, y)
    assert np.allclose(hdg_north, 0.0)

    # Moving East: x increases, y=0 -> 90 deg
    x = np.linspace(0, 100, 10)
    y = np.zeros(10)
    hdg_east = compute_gps_heading(x, y)
    assert np.allclose(hdg_east, 90.0)

    # Moving South: x=0, y decreases -> 180 deg
    x = np.zeros(10)
    y = np.linspace(100, 0, 10)
    hdg_south = compute_gps_heading(x, y)
    assert np.allclose(hdg_south, 180.0)




# ---------------------------------------------------------------------------
# Empty channels guard
# ---------------------------------------------------------------------------
def test_empty_log_guards():
    log = DataLog()
    assert log.duration() == 0.0
    assert log.start() == 0.0
    assert log.end() == 0.0
    log.resample(20)  # must not crash


# ---------------------------------------------------------------------------
# ZOH interpolation helper
# ---------------------------------------------------------------------------
def test_interp_zoh_discrete():
    from data_log import _interp_zoh
    src_t = np.array([0.0, 1.0, 2.0])
    src_v = np.array([2.0, 3.0, 1.0])
    new_t = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    result = _interp_zoh(new_t, src_t, src_v)
    expected = np.array([2.0, 2.0, 3.0, 3.0, 1.0, 1.0])
    assert np.array_equal(result, expected), f"Got {result}, expected {expected}"


# ---------------------------------------------------------------------------
# CRC check - resample does not alter discrete values
# ---------------------------------------------------------------------------
def test_gear_preserved_after_resample():
    log = DataLog()
    log.add_channel("Gear", "", float, 0)
    n = 100
    for i in range(n):
        gear = 2 if i < n // 3 else (3 if i < 2 * n // 3 else 4)
        log.channels["Gear"].messages.append(
            type("M", (), {"timestamp": float(i) * 0.05, "value": float(gear)})()
        )
    log.resample(20)
    for m in log.channels["Gear"].messages:
        assert float(m.value).is_integer(), f"Non-integer gear value: {m.value}"


def test_auto_frequency_detection():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))
    freq = log.detect_native_frequency()
    assert freq > 0, f"Invalid auto frequency: {freq}"
    resampled_f = log.resample("auto")
    assert resampled_f == freq
def test_sector_beacons_detection():
    log = DataLog()
    log.traps = [
        {"name": "Start/Finish", "lat": 37.0, "lon": -122.0, "type": 3},
        {"name": "Split 1", "lat": 37.0001, "lon": -122.0001, "type": 4},
    ]
    log.add_channel("GPS Latitude", "deg", float, 7)
    log.add_channel("GPS Longitude", "deg", float, 7)
    log.channels["GPS Latitude"].messages = [
        Message(0.0, 37.0002), Message(1.0, 37.0), Message(2.0, 37.0002),
        Message(10.0, 37.0003), Message(11.0, 37.0001), Message(12.0, 37.0003)
    ]
    log.channels["GPS Longitude"].messages = [
        Message(0.0, -122.0002), Message(1.0, -122.0), Message(2.0, -122.0002),
        Message(10.0, -122.0003), Message(11.0, -122.0001), Message(12.0, -122.0003)
    ]
    beacons = log.detect_beacons(min_speed_kmh=0.0, min_time_sec=0.0)
    assert len(beacons) >= 2
    assert beacons[0][1] in ("Start/Finish", "Split 1")


# ---------------------------------------------------------------------------
# Regression: from_csv_log must not delete channels right of a non-numeric column
# (previously the channel_dict index remapping broke column lookup)
# ---------------------------------------------------------------------------
def test_csv_middle_non_numeric_column_keeps_right_columns():
    log = DataLog()
    lines = [
        "time,colA,colB,colC\n",
        "0.0,1.0,XXX,100.0\n",
        "0.1,2.0,YYY,200.0\n",
        "0.2,3.0,ZZZ,300.0\n",
    ]
    log.from_csv_log(lines)
    assert "colA" in log.channels
    assert "colC" in log.channels
    assert "colB" not in log.channels
    colC = log.channels["colC"].messages
    assert len(colC) == 3, f"colC messages: {len(colC)}"
    assert colC[-1].value == 300.0


# ---------------------------------------------------------------------------
# AIM XRK / XRZ
# ---------------------------------------------------------------------------
def test_xrk_smoke():
    _import_or_skip("libxrk")

    log = DataLog()
    log.from_xrk_log(os.path.join(EXAMPLES, "aim_sample.xrk"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.duration() > 0
    _dedup_channels(log)


def test_xrk_units_time_origin_and_datetime():
    _import_or_skip("libxrk")

    log = DataLog()
    log.from_xrk_log(os.path.join(EXAMPLES, "aim_sample.xrk"))

    speed = log.channels["Ground Speed"]
    rpm = log.channels["Engine RPM"]
    gps = log.channels["GPS Latitude"]
    assert speed.units == "km/h"
    assert max(speed.values) > 100.0, "XRK m/s speed was labelled km/h without conversion"
    assert rpm.start() == 0.0
    assert 4.7 < gps.start() < 4.8, "Per-channel time origins must not be collapsed to zero"
    assert log.datetime.strftime("%Y-%m-%d %H:%M:%S") == "2016-01-23 12:09:04"


# ---------------------------------------------------------------------------
# Garmin FIT
# ---------------------------------------------------------------------------
def test_fit_smoke():
    _import_or_skip("fitparse")

    log = DataLog()
    log.from_fit_log(os.path.join(EXAMPLES, "garmin_sample.fit"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.duration() > 0
    _dedup_channels(log)


def test_fit_unique_timestamps_finite_math_and_lap_boundaries():
    _import_or_skip("fitparse")

    log = DataLog()
    log.from_fit_log(os.path.join(EXAMPLES, "garmin_sample.fit"))
    speed = log.channels["Ground Speed"]
    assert np.all(np.diff(speed.timestamps) > 0), "FIT timestamps must be strictly increasing"

    log.calculate_math_channels()
    for name in ("CG Accel Longitudinal", "CG Accel Long Smooth"):
        assert np.all(np.isfinite(log.channels[name].values)), f"{name} contains NaN/Inf"

    laps = log.laps_info["laps"]
    assert len(log.laps_info["beacons"]) == len(laps) + 1
    for previous, current in zip(laps, laps[1:]):
        assert previous["end_time"] == current["start_time"]


def test_gear_derivation_aligns_channels_by_timestamp():
    from processing.math_channels import derive_gear_from_rpm_speed

    log = DataLog()
    log.add_channel("Engine RPM", "rpm", float, 0)
    log.channels["Engine RPM"].messages = [
        Message(i / 10.0, 3000.0) for i in range(21)
    ]
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].messages = [
        Message(0.0, 30.0), Message(1.0, 30.0),
        Message(1.0, 30.0), Message(2.0, 30.0)
    ]

    derive_gear_from_rpm_speed(log)
    gear = log.channels["Gear"]
    assert np.array_equal(gear.timestamps, np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(gear.values, np.array([2.0, 2.0, 2.0]))


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def test_csv_export():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))
    log.resample(20)

    import tempfile
    from exporters.csv_export import write_csv

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        assert write_csv(log, tmp_path)
        with open(tmp_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 2, "CSV should have header + data rows"
        header = lines[0].strip().split(",")
        assert "Time (s)" in header
        assert len(header) >= 2, "CSV should have at least one channel column"
    finally:
        os.remove(tmp_path)


def test_csv_export_uses_timestamp_union_and_handles_empty_channels():
    from exporters.csv_export import write_csv

    log = DataLog()
    log.add_channel("Engine RPM", "rpm", float, 0)
    log.channels["Engine RPM"].messages = [
        Message(i / 100.0, 1000.0 + i) for i in range(101)
    ]
    for name, base in (("GPS Latitude", 37.0), ("GPS Longitude", -122.0)):
        log.add_channel(name, "deg", float, 7)
        log.channels[name].messages = [
            Message(i / 10.0, base + i * 1e-5) for i in range(11)
        ]
    log.add_channel("Empty", "", float, 0)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        assert write_csv(log, tmp_path)
        with open(tmp_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) - 1 == 101
        assert rows[0][:3] == ["Time (s)", "GPS Latitude", "GPS Longitude"]
        assert all(row[-1] == "" for row in rows[1:])
    finally:
        os.remove(tmp_path)


def test_ca9_alignment_defaults_to_no_output():
    from tools import align_ca9_mesh

    parser = align_ca9_mesh.build_arg_parser()
    args = parser.parse_args([])
    assert args.output is None
    assert align_ca9_mesh.resolve_output_path(args) is None

    args = parser.parse_args(["--merged", "same.ld", "--output", "same.ld"])
    try:
        align_ca9_mesh.resolve_output_path(args)
    except ValueError:
        pass
    else:
        raise AssertionError("in-place CA-9 replacement must require --force")


def test_cli_ld_ldx_roundtrip_and_overwrite_guard():
    from ldparser.ldparser import ldData

    cli = os.path.join(ROOT, "motec_log_generator.py")
    source = os.path.join(EXAMPLES, "aim_solo_sample.csv")
    with tempfile.TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, "roundtrip.ld")
        command = [
            sys.executable, cli, source, "RACECHRONO",
            "--output", output,
        ]

        first = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        assert first.returncode == 0, first.stdout + first.stderr

        ldx_path = os.path.splitext(output)[0] + ".ldx"
        assert os.path.isfile(output)
        assert os.path.isfile(ldx_path)

        parsed = ldData.fromfile(output)
        assert parsed.channs
        for channel in parsed.channs:
            values = channel.data
            assert len(values) == channel.data_len
        root = ET.parse(ldx_path).getroot()
        assert root.tag == "LDXFile"

        before = {path: _sha256(path) for path in (output, ldx_path)}
        blocked = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        assert blocked.returncode != 0
        assert "already exists" in (blocked.stdout + blocked.stderr).lower()
        after = {path: _sha256(path) for path in (output, ldx_path)}
        assert after == before

        profile_path = os.path.join(ROOT, "vehicle_profiles", "gr86.json")
        forced = subprocess.run(
            command + ["--force", "--kinematics", "--vehicle-profile", profile_path],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert forced.returncode == 0, forced.stdout + forced.stderr
        parsed = ldData.fromfile(output)
        assert parsed.head.vehicleid == "Toyota GR86"
        assert "Understeer Index" in [channel.name for channel in parsed.channs]


def test_atomic_output_verification_failure_preserves_existing_files():
    from core.output import atomic_write_motec_pair, ensure_output_targets
    from motec_log import MotecLog

    log = DataLog()
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].set_samples([0.0, 1.0], [10.0, 20.0])
    motec = MotecLog()
    motec.initialize()
    motec.add_all_channels(log)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ld_path = os.path.join(tmp_dir, "existing.ld")
        ldx_path = os.path.join(tmp_dir, "existing.ldx")
        with open(ld_path, "wb") as output:
            output.write(b"old ld")
        with open(ldx_path, "wb") as output:
            output.write(b"old ldx")

        with patch("core.output.verify_motec_pair", side_effect=RuntimeError("bad staged file")):
            try:
                atomic_write_motec_pair(motec, log, ld_path, ldx_path)
            except RuntimeError as exc:
                assert "bad staged file" in str(exc)
            else:
                raise AssertionError("verification failure must abort output replacement")

        with open(ld_path, "rb") as output:
            assert output.read() == b"old ld"
        with open(ldx_path, "rb") as output:
            assert output.read() == b"old ldx"
        assert not [name for name in os.listdir(tmp_dir) if name.startswith(".motec_")]

        real_replace = os.replace
        replace_calls = 0

        def fail_second_replace(source, target):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 2:
                raise PermissionError("second target locked")
            return real_replace(source, target)

        with patch("core.output.os.replace", side_effect=fail_second_replace):
            try:
                atomic_write_motec_pair(motec, log, ld_path, ldx_path)
            except PermissionError as exc:
                assert "second target locked" in str(exc)
            else:
                raise AssertionError("second replace failure must abort the pair commit")

        with open(ld_path, "rb") as output:
            assert output.read() == b"old ld"
        with open(ldx_path, "rb") as output:
            assert output.read() == b"old ldx"
        assert not [name for name in os.listdir(tmp_dir) if name.startswith(".motec_")]

        try:
            ensure_output_targets([ld_path], force=True, source_path=ld_path)
        except ValueError as exc:
            assert "input file" in str(exc)
        else:
            raise AssertionError("--force must never replace the source input")


def test_vehicle_profile_configures_kinematics():
    from motec_log_generator import load_vehicle_profile
    from processing.vehicle_profile import validate_vehicle_profile
    from processing.math_channels import calculate_kinematics

    profile_doc = {
        "name": "test car",
        "kinematics": {
            "steering_ratio": 10.0,
            "wheelbase_m": 3.0,
            "cg_to_front_axle_m": 1.4,
            "cg_to_rear_axle_m": 1.6,
            "lateral_velocity_tau_s": 1.0,
        },
        "gear_ratio_thresholds": [120, 80, 60, 45, 35, 22],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     encoding="utf-8") as tmp:
        json.dump(profile_doc, tmp)
        profile_path = tmp.name
    try:
        profile = load_vehicle_profile(profile_path)
    finally:
        os.remove(profile_path)

    log = DataLog()
    samples = [0.0, 1.0, 2.0]
    for name, unit, values in (
        ("Ground Speed", "km/h", [72.0, 72.0, 72.0]),
        ("CG Accel Lateral", "G", [0.5, 0.5, 0.5]),
        ("Chassis Yaw Rate", "deg/s", [10.0, 10.0, 10.0]),
        ("Steering Angle", "deg", [100.0, 100.0, 100.0]),
    ):
        log.add_channel(name, unit, float, 2)
        log.channels[name].set_samples(samples, values)

    calculate_kinematics(log, parameters=profile["kinematics"])
    assert "Understeer Index" in log.channels
    expected = 10.0 - np.degrees(3.0 * np.radians(-10.0) / 20.0)
    assert abs(log.channels["Understeer Index"].values[-1] - expected) < 1e-9

    try:
        validate_vehicle_profile({"kinematics": {"wheelbase_m": 3.0}})
    except ValueError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("incomplete kinematics profile must be rejected")


def test_csv_auxiliary_export_does_not_replace_source():
    from motec_log_generator import _csv_output_path

    source = os.path.join("logs", "session.csv")
    assert _csv_output_path(os.path.join("logs", "session.ld"), source).endswith(
        "session_export.csv"
    )



if __name__ == "__main__":
    # Run all test_* functions directly without pytest
    import traceback
    g = globals()
    tests = sorted(name for name in g if name.startswith("test_"))
    passed = 0
    failed = 0
    skipped = 0
    for name in tests:
        fn = g[name]
        if not callable(fn) or name.startswith("pytest"):
            continue
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except SkipTest as e:
            print(f"  SKIP {name}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)
