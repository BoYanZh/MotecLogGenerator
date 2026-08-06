"""Smoke and regression tests using the example/ input files."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_log import DataLog, Message

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _read_lines(filename):
    path = os.path.join(EXAMPLES, filename)
    with open(path) as f:
        return f.readlines()


def _dedup_channels(log):
    """MoTeC doesn't allow duplicate channel names, validate this invariant."""
    assert len(log.channels) == len(set(log.channels)), f"Duplicate channel names: {list(log.channels)}"


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def test_csv_smoke():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))
    assert len(log.channels) >= 3, f"Too few channels: {len(log.channels)}"
    assert log.duration() > 0
    _dedup_channels(log)


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
    pytest = pytest_if_available()
    if pytest is None:
        return

    try:
        import cantools
    except ImportError:
        pytest.skip("cantools not installed")

    log = DataLog()
    dbc_path = os.path.join(EXAMPLES, "sample_can_spec.dbc")
    can_db = cantools.database.load_file(dbc_path)
    log.from_can_log(_read_lines("can_sample.log"), can_db)
    assert len(log.channels) >= 1, f"Too few channels: {len(log.channels)}"
    assert log.duration() > 0
    _dedup_channels(log)


def test_can_resample():
    pytest = pytest_if_available()
    if pytest is None:
        return
    try:
        import cantools
    except ImportError:
        pytest.skip("cantools not installed")

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
# CRC check — resample does not alter discrete values
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


def pytest_if_available():
    """Return the pytest module if available, None otherwise (for standalone runs)."""
    try:
        import pytest
        return pytest
    except ImportError:
        return None


if __name__ == "__main__":
    # Run all test_* functions directly without pytest
    import traceback
    g = globals()
    tests = sorted(name for name in g if name.startswith("test_"))
    passed = 0
    failed = 0
    for name in tests:
        fn = g[name]
        if not callable(fn) or name.startswith("pytest"):
            continue
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
