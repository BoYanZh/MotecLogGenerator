#!/usr/bin/env python3
"""
Corner-by-Corner Time Loss Analysis Utility.

Compares two laps (from .rcz, .ld, or .csv files) on a spatial distance axis,
computing corner-by-corner time delta, apex speed, braking intensity, and
throttle application differences with priority-ranked coaching output.

Lap selection:
  - By default, the FASTEST lap from each session is used.
  - Use --ref_lap / --target_lap to specify a particular lap time (MM:SS.mmm).

Supported input formats: .rcz  .ld  .csv (RaceChrono CSV export)

Usage:
    # Auto best laps:
    python tools/analyze_corner_time_loss.py ref.rcz target.rcz

    # Specific laps by time:
    python tools/analyze_corner_time_loss.py ref.rcz target.rcz \\
        --ref_lap 2:07.710 --target_lap 2:16.450

    # Batch: compare all .ld files in a dir, fastest as benchmark:
    python tools/analyze_corner_time_loss.py --dir data/exported
"""

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ldparser.ldparser import ldData
from data_log import DataLog

# ────────────────────────────────────────────────────────────────────────────
# Corner segment definitions (distance from start/finish in metres)
# ────────────────────────────────────────────────────────────────────────────
CORNER_PRESETS = {
    "thunderhill_east_bypass": [
        {"name": "Turn 1 (Fast Left)",        "start_m":  100, "end_m":  450},
        {"name": "Turn 2 (Long Sweeper)",     "start_m":  450, "end_m":  858},
        {"name": "Turn 3 (Off-Camber)",       "start_m":  858, "end_m": 1200},
        {"name": "Turn 4/5 (Cyclone Hill)",   "start_m": 1200, "end_m": 1700},
        {"name": "Turn 6/7 (S-Turn)",         "start_m": 1700, "end_m": 2000},
        {"name": "Turn 8 (Bypass Sweeper)",   "start_m": 2000, "end_m": 2400},
        {"name": "Turn 9 (Uphill Crest)",     "start_m": 2400, "end_m": 2800},
        {"name": "Turn 10/11 (Chicane)",      "start_m": 2800, "end_m": 3400},
        {"name": "Turn 14/15 (Final Corner)", "start_m": 3800, "end_m": 4200},
        {"name": "Front Straight (Full WOT)", "start_m": 4200, "end_m": 4540},
    ],
}
DEFAULT_PRESET = "thunderhill_east_bypass"


# ────────────────────────────────────────────────────────────────────────────
# Lap time parsing helpers
# ────────────────────────────────────────────────────────────────────────────

def parse_lap_time_str(s):
    """'2:07.710' or '127.710' → seconds (float)."""
    if not s:
        return None
    try:
        if ":" in s:
            m, rest = s.split(":", 1)
            return int(m) * 60.0 + float(rest)
        return float(s)
    except ValueError:
        return None


def fmt_lap_time(sec):
    if sec is None or sec <= 0:
        return "N/A"
    m = int(sec // 60)
    return "%d:%06.3f" % (m, sec % 60)


# ────────────────────────────────────────────────────────────────────────────
# Beacon / lap boundary extraction
# ────────────────────────────────────────────────────────────────────────────

def _beacons_from_ldx(ldx_path):
    """Return sorted list of Start/Finish beacon timestamps (seconds)."""
    if not os.path.isfile(ldx_path):
        return []
    try:
        root = ET.parse(ldx_path).getroot()
        beacons = []
        for m in root.findall(".//Marker"):
            t = m.get("Time")
            if t:
                beacons.append(float(t) / 1e6)
        return sorted(beacons)
    except Exception:
        return []


def _find_lap_windows(beacons, min_lap=60.0, max_lap=360.0):
    """Given a list of beacon timestamps, return [(t_start, t_end, duration)] for valid laps."""
    windows = []
    for i in range(len(beacons) - 1):
        dur = beacons[i + 1] - beacons[i]
        if min_lap <= dur <= max_lap:
            windows.append((beacons[i], beacons[i + 1], dur))
    return windows


def _select_lap_window(windows, target_sec=None, tol=2.0):
    """
    Pick a lap window.
    If target_sec is given, pick the window whose duration is closest (within tol s).
    Otherwise pick the fastest.
    Returns (t_start, t_end, duration) or None.
    """
    if not windows:
        return None
    if target_sec is not None:
        candidates = [(abs(w[2] - target_sec), w) for w in windows]
        candidates.sort(key=lambda x: x[0])
        best_diff, best = candidates[0]
        if best_diff > tol:
            print("WARNING: Closest lap found is %.3fs away from target %.3fs" % (best_diff, target_sec))
        return best
    # fastest
    return min(windows, key=lambda w: w[2])


# ────────────────────────────────────────────────────────────────────────────
# Core: load a single lap from any supported file format
# ────────────────────────────────────────────────────────────────────────────

_CHANNELS = ["Ground Speed", "CG Accel Lateral", "CG Accel Longitudinal",
             "Throttle Pos", "Brake Press"]


def _spatial_slice(spd_kmh, brk, thr, lat, lng, freq, t_start=None, t_end=None, step_m=1.0):
    """
    Convert a time-domain speed trace (+ optional channel arrays) to a 1 m spatial grid.
    If t_start/t_end are given, slice that window first.
    Returns dict with s_grid, v_kmh, t_grid, brk, thr, lat, lng, total_dist_m, lap_time_sec.
    """
    dt = 1.0 / freq
    n = len(spd_kmh)
    t_arr = np.arange(n) * dt

    if t_start is not None and t_end is not None:
        mask = (t_arr >= t_start) & (t_arr <= t_end)
        spd_kmh = spd_kmh[mask]
        brk     = brk[mask]     if brk is not None else None
        thr     = thr[mask]     if thr is not None else None
        lat     = lat[mask]     if lat is not None else None
        lng     = lng[mask]     if lng is not None else None
        lap_time = t_end - t_start
    else:
        lap_time = n * dt

    spd_ms = np.maximum(spd_kmh / 3.6, 0.5)
    dist = np.cumsum(spd_ms * dt)
    total_dist = dist[-1]

    s = np.arange(0.0, total_dist, step_m)

    def interp_or_zeros(arr):
        if arr is None:
            return np.zeros(len(s))
        return np.interp(s, dist, arr)

    v_s   = np.interp(s, dist, spd_kmh)
    t_s   = np.cumsum(step_m / np.maximum(np.interp(s, dist, spd_ms), 0.5))
    brk_s = interp_or_zeros(brk)
    thr_s = interp_or_zeros(thr)
    lat_s = interp_or_zeros(lat)
    lng_s = interp_or_zeros(lng)

    return {
        "s": s, "v": v_s, "t": t_s,
        "brk": brk_s, "thr": thr_s, "lat": lat_s, "lng": lng_s,
        "total_dist_m": total_dist,
        "lap_time_sec": lap_time,
    }


def _load_from_data_log(dl, target_sec=None, tol=2.0, verbose=False):
    """
    Extract lap spatial data from a DataLog object (already loaded + math channels computed).
    Writes a temporary .ld/.ldx to disk so we can use beacon-based lap detection.
    """
    def get(name):
        ch = dl.channels.get(name)
        if ch is None:
            return None, None
        msgs = ch.messages
        if not msgs:
            return None, None
        times = np.array([m.timestamp for m in msgs])
        vals  = np.array([m.value     for m in msgs])
        return times, vals

    spd_t, spd_v = get("Ground Speed")
    if spd_v is None or len(spd_v) < 50:
        return None

    # Build uniform time grid
    dt_arr = np.diff(spd_t)
    dt_arr = dt_arr[dt_arr > 0]
    freq = 1.0 / np.median(dt_arr) if len(dt_arr) else 20.0
    dt = 1.0 / freq
    t_uniform = np.arange(spd_t[0], spd_t[-1], dt)
    spd_uniform = np.interp(t_uniform, spd_t, spd_v)

    def resample(name):
        tt, vv = get(name)
        if tt is None:
            return None
        return np.interp(t_uniform, tt, vv)

    brk = resample("Brake Press")
    thr = resample("Throttle Pos")
    lat = resample("CG Accel Lateral")
    lng = resample("CG Accel Longitudinal")

    # ── Strategy 1: Lap Number channel ───────────────────────────────────────
    lap_t, lap_v = get("Lap Number")
    windows = []
    if lap_v is not None and len(lap_v) > 0:
        lap_uniform = np.interp(t_uniform, lap_t, lap_v).round().astype(int)
        for lap_num in np.unique(lap_uniform):
            if lap_num < 1:
                continue
            idx = np.where(lap_uniform == lap_num)[0]
            if len(idx) < 10:
                continue
            t_s_abs = t_uniform[idx[0]]
            t_e_abs = t_uniform[idx[-1]]
            dur = t_e_abs - t_s_abs
            if 60.0 <= dur <= 360.0:
                windows.append((t_s_abs, t_e_abs, dur))

    # ── Strategy 2: export temp .ld/.ldx, then parse ldx beacons ─────────────
    if not windows:
        import tempfile
        try:
            from motec_log import MotecLog
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_ld  = os.path.join(tmpdir, "tmp_session.ld")
                tmp_ldx = tmp_ld.replace(".ld", ".ldx")
                ml = MotecLog()
                ml.initialize()
                ml.add_all_channels(dl)
                ml.write(tmp_ld)
                beacons_raw = dl.detect_beacons()
                ml.write_ldx(tmp_ldx, getattr(dl, "laps_info", None), beacons=beacons_raw)
                ldx_beacons = _beacons_from_ldx(tmp_ldx)
                windows = _find_lap_windows(ldx_beacons)
                # ldx beacons are session-relative (start from 0); t_uniform is absolute.
                t_offset = t_uniform[0]
                windows = [(t_offset + w[0], t_offset + w[1], w[2]) for w in windows]
                if verbose:
                    print("  Found %d lap windows from temp .ldx export" % len(windows))
        except Exception as e:
            if verbose:
                print("  Temp .ldx export failed: %s" % e)



    # ── Strategy 3: speed-reset heuristic (distance drop) ────────────────────
    if not windows:
        dist_t, dist_v = get("Distance on GPS Speed")
        if dist_v is not None and len(dist_v) > 10:
            dist_uniform = np.interp(t_uniform, dist_t, dist_v)
            ddist = np.diff(dist_uniform)
            reset_idx = np.where(ddist < -50)[0]  # distance reset = new lap
            lap_starts = np.concatenate([[0], reset_idx + 1])
            lap_ends   = np.concatenate([reset_idx, [len(t_uniform) - 1]])
            for s_i, e_i in zip(lap_starts, lap_ends):
                dur = t_uniform[e_i] - t_uniform[s_i]
                if 60.0 <= dur <= 360.0:
                    windows.append((t_uniform[s_i], t_uniform[e_i], dur))

    # ── Fallback: whole trace as single lap ───────────────────────────────────
    if not windows:
        lap_time = t_uniform[-1] - t_uniform[0]
        if verbose:
            print("  No lap boundaries found; treating whole trace as single lap (%.1fs)" % lap_time)
        result = _spatial_slice(spd_uniform, brk, thr, lat, lng, freq, lap_time=lap_time)
        return result

    chosen = _select_lap_window(windows, target_sec, tol)
    if not chosen:
        return None

    t_s, t_e, dur = chosen
    if verbose:
        print("  Selected lap: %s  (%.3fs)  [%.3f → %.3f]" % (fmt_lap_time(dur), dur, t_s, t_e))

    mask = (t_uniform >= t_s) & (t_uniform <= t_e)
    return _spatial_slice(
        spd_uniform[mask],
        brk[mask] if brk is not None else None,
        thr[mask] if thr is not None else None,
        lat[mask] if lat is not None else None,
        lng[mask] if lng is not None else None,
        freq, lap_time=dur,
    )




def _load_from_ld(ld_path, target_sec=None, tol=2.0, verbose=False):
    """Extract lap spatial data from a MoTeC .ld file + companion .ldx."""
    try:
        ld = ldData.fromfile(ld_path)
    except Exception as e:
        print("ERROR reading %s: %s" % (ld_path, e))
        return None

    def get_ch(name):
        for c in ld.channs:
            if c.name == name:
                return c.data
        return None

    spd = get_ch("Ground Speed")
    if spd is None or len(spd) < 50:
        return None

    freq = ld.channs[0].freq if ld.channs else 20.0
    brk  = get_ch("Brake Press")
    thr  = get_ch("Throttle Pos")
    lat  = get_ch("CG Accel Lateral")
    lng  = get_ch("CG Accel Longitudinal")

    driver  = getattr(ld.head, "driver",    "") or ""
    vehicle = getattr(ld.head, "vehicleid", "") or ""
    venue   = getattr(ld.head, "venue",     "") or ""

    # Parse .ldx beacons
    ldx_path = os.path.splitext(ld_path)[0] + ".ldx"
    beacons  = _beacons_from_ldx(ldx_path)
    windows  = _find_lap_windows(beacons)

    if not windows:
        if verbose:
            print("  No valid lap windows from .ldx; treating whole file as one lap.")
        result = _spatial_slice(spd, brk, thr, lat, lng, freq)
        result.update({"driver": driver, "vehicle": vehicle, "venue": venue,
                        "file": os.path.basename(ld_path)})
        return result

    chosen = _select_lap_window(windows, target_sec, tol)
    if not chosen:
        return None
    t_s, t_e, dur = chosen
    if verbose:
        print("  Selected lap: %s  (%.3fs)  [%.3f → %.3f]" % (fmt_lap_time(dur), dur, t_s, t_e))

    result = _spatial_slice(spd, brk, thr, lat, lng, freq, t_start=t_s, t_end=t_e)
    result.update({"driver": driver, "vehicle": vehicle, "venue": venue,
                    "file": os.path.basename(ld_path)})
    return result


# monkey-patch lap_time into _spatial_slice when called without t_start/t_end
def _spatial_slice(spd_kmh, brk, thr, lat, lng, freq,
                   t_start=None, t_end=None, lap_time=None, step_m=1.0):
    dt = 1.0 / freq
    n = len(spd_kmh)

    if t_start is not None and t_end is not None:
        t_arr = np.arange(n) * dt
        mask = (t_arr >= t_start) & (t_arr <= t_end)
        spd_kmh = spd_kmh[mask]
        brk     = brk[mask]  if brk is not None else None
        thr     = thr[mask]  if thr is not None else None
        lat     = lat[mask]  if lat is not None else None
        lng     = lng[mask]  if lng is not None else None
        lap_time = t_end - t_start

    if lap_time is None:
        lap_time = len(spd_kmh) * dt

    spd_ms = np.maximum(spd_kmh / 3.6, 0.5)
    dist   = np.cumsum(spd_ms * dt)
    total  = dist[-1]
    s      = np.arange(0.0, total, step_m)

    def interp_or_zeros(arr):
        return np.interp(s, dist, arr) if arr is not None else np.zeros(len(s))

    return {
        "s": s, "v": np.interp(s, dist, spd_kmh),
        "t": np.cumsum(step_m / np.maximum(np.interp(s, dist, spd_ms), 0.5)),
        "brk": interp_or_zeros(brk), "thr": interp_or_zeros(thr),
        "lat": interp_or_zeros(lat), "lng": interp_or_zeros(lng),
        "total_dist_m": total, "lap_time_sec": lap_time,
    }


def load_lap(file_path, target_sec=None, tol=2.0, verbose=False):
    """
    Universal loader. Accepts .rcz, .ld, or .csv.
    Returns a spatial lap dict or None on failure.
    """
    ext = os.path.splitext(file_path)[1].lower()
    label = os.path.basename(file_path)
    if verbose:
        print("Loading: %s  (target lap: %s)" % (label,
              fmt_lap_time(target_sec) if target_sec else "fastest"))

    # ── .ld ──────────────────────────────────────────────────────────────────
    if ext == ".ld":
        result = _load_from_ld(file_path, target_sec, tol, verbose)
        if result:
            result.setdefault("file", label)
        return result

    # ── .rcz or .csv → load via DataLog ──────────────────────────────────────
    dl = DataLog()

    if ext == ".rcz":
        # For multi-stint RCZ files, scan all stints for the target lap
        import zipfile, json
        stints = [None]  # None = default (stint 0)
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                all_names = z.namelist()
            resume_idxs = sorted(set(
                int(n.split("/")[0].replace("resume_", ""))
                for n in all_names if n.startswith("resume_")
            ))
            if resume_idxs:
                stints = [0] + resume_idxs
        except Exception:
            pass

        best_result = None
        best_diff   = float("inf")

        for stint in stints:
            dl2 = DataLog()
            try:
                dl2.from_rcz_log(file_path, target_stint=stint)
                dl2.calculate_math_channels(g_source="auto")
            except Exception as e:
                if verbose:
                    print("  Stint %s failed: %s" % (stint, e))
                continue

            r = _load_from_data_log(dl2, target_sec, tol, verbose)
            if r is None:
                continue
            if target_sec is None:
                # Collect fastest across all stints
                if best_result is None or r["lap_time_sec"] < best_result["lap_time_sec"]:
                    best_result = r
            else:
                diff = abs(r["lap_time_sec"] - target_sec)
                if diff < best_diff:
                    best_diff   = diff
                    best_result = r

        result = best_result

    elif ext == ".csv":
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            print("ERROR reading CSV %s: %s" % (file_path, e))
            return None
        # Auto-detect CSV sub-format
        sample = "".join(lines[:20])
        try:
            if any(kw in sample for kw in ("RaceChrono", "RaceStudio", "Solo", "GPS_LatAcc",
                                            "AiM", "Sample Rate")):
                dl.from_racechrono_log(lines, target_lap=None)  # load all laps
            else:
                dl.from_csv_log(lines)
            dl.calculate_math_channels(g_source="auto")
        except Exception as e:
            print("ERROR loading CSV %s: %s" % (file_path, e))
            return None
        result = _load_from_data_log(dl, target_sec, tol, verbose)

    else:
        print("ERROR: Unsupported file extension '%s'" % ext)
        return None

    if result:
        result.setdefault("driver",  "")
        result.setdefault("vehicle", "")
        result.setdefault("venue",   "")
        result["file"] = label
    return result


# ────────────────────────────────────────────────────────────────────────────
# Analysis & reporting
# ────────────────────────────────────────────────────────────────────────────

def _diag(time_loss, vmin_diff, ref_v_entry, tgt_v_entry, ref_v_exit, tgt_v_exit,
          ref_brk_max, tgt_brk_max):
    """Classify primary cause of time loss in a corner."""
    if time_loss <= 0.05:
        return "Gained Time" if time_loss < -0.05 else "Equal Pace"
    if vmin_diff < -3.0 and tgt_brk_max < ref_brk_max * 0.75:
        return "Timid Entry / Insufficient Braking"
    if vmin_diff < -3.0:
        return "Low Apex Speed / Over-braking"
    if ref_v_entry - tgt_v_entry > 4.0:
        return "Early Braking on Entry"
    if ref_v_exit - tgt_v_exit > 4.0:
        return "Late / Hesitant Throttle"
    return "General Loss"


def analyze_and_report(ref, tgt, corners, label_ref="REF", label_tgt="TGT", verbose=False):
    max_dist = min(ref["total_dist_m"], tgt["total_dist_m"])
    s = np.arange(0.0, max_dist, 1.0)

    def resamp(lap, key):
        return np.interp(s, lap["s"], lap[key])

    r_v   = resamp(ref, "v");   t_v   = resamp(tgt, "v")
    r_t   = resamp(ref, "t");   t_t   = resamp(tgt, "t")
    r_brk = resamp(ref, "brk"); t_brk = resamp(tgt, "brk")
    r_thr = resamp(ref, "thr"); t_thr = resamp(tgt, "thr")
    r_lat = resamp(ref, "lat"); t_lat = resamp(tgt, "lat")

    dt = t_t - r_t
    total_delta = dt[-1]

    W = 114
    print("\n" + "=" * W)
    print("  CORNER-BY-CORNER TIME LOSS & DELTA ANALYSIS")
    print("  Reference  : %-40s  %s" % (ref["file"], fmt_lap_time(ref["lap_time_sec"])))
    print("  Target     : %-40s  %s" % (tgt["file"], fmt_lap_time(tgt["lap_time_sec"])))
    print("  Total Δt   : %+.3fs  (%s)"
          % (total_delta, "SLOWER" if total_delta > 0 else "FASTER"))
    print("=" * W)

    # Whole-lap stats
    print()
    print("  WHOLE-LAP STATISTICS")
    print("  %-22s  %10s  %10s  %10s" % ("", "Reference", "Target", "Delta"))
    print("  " + "-" * 58)

    def row(label, r_val, t_val, unit="", fmt="%.1f"):
        pat = "  %-22s  " + fmt + " %-4s  " + fmt + " %-4s  " + ("+" if (t_val-r_val) >= 0 else "") + fmt + " %-4s"
        print(pat % (label, r_val, unit, t_val, unit, t_val - r_val, unit))

    row("Avg Speed",        r_v.mean(),                t_v.mean(),                "km/h")
    row("Max Speed",        r_v.max(),                 t_v.max(),                 "km/h")
    row("Peak |Lat G|",     np.abs(r_lat).max(),       np.abs(t_lat).max(),       "G",    "%.3f")
    row("Mean |Lat G|",     np.abs(r_lat).mean(),      np.abs(t_lat).mean(),      "G",    "%.3f")
    row("WOT % (thr>95)",   (r_thr > 95).mean()*100,  (t_thr > 95).mean()*100,  "%",    "%.1f")
    row("Braking % (>200)", (r_brk > 200).mean()*100, (t_brk > 200).mean()*100, "%",    "%.1f")
    row("Peak Brake",       r_brk.max(),               t_brk.max(),               "kPa",  "%.0f")

    print()
    print("  CORNER-BY-CORNER BREAKDOWN")
    hdr = "  %-22s | %8s | %7s %7s %7s | %7s %7s | %7s %7s | %-28s"
    print(hdr % ("Corner", "Δt (s)",
                 "Vmin R", "Vmin T", "ΔVmin",
                 "Brk R", "Brk T",
                 "WOT R%", "WOT T%",
                 "Primary Cause"))
    print("  " + "-" * (W - 2))

    priority_list = []  # (time_loss, corner_name, diag, details)

    for c in corners:
        s0, s1 = c["start_m"], min(c["end_m"], max_dist)
        if s0 >= max_dist:
            continue
        idx = np.where((s >= s0) & (s <= s1))[0]
        if len(idx) < 5:
            continue

        tl = (t_t[idx[-1]] - t_t[idx[0]]) - (r_t[idx[-1]] - r_t[idx[0]])

        r_vmin = r_v[idx].min();  t_vmin = t_v[idx].min()
        r_brk_max = r_brk[idx].max(); t_brk_max = t_brk[idx].max()
        r_wot = (r_thr[idx] > 95).mean() * 100
        t_wot = (t_thr[idx] > 95).mean() * 100

        # entry = first 25%, exit = last 25%
        q = max(1, len(idx) // 4)
        r_entry = r_v[idx[:q]].mean(); t_entry = t_v[idx[:q]].mean()
        r_exit  = r_v[idx[-q:]].mean(); t_exit = t_v[idx[-q:]].mean()

        diag = _diag(tl, t_vmin - r_vmin, r_entry, t_entry, r_exit, t_exit,
                     r_brk_max, t_brk_max)

        tl_str = "%+.3f" % tl
        flag = " ◄" if tl > 0.3 else ""
        print("  %-22s | %8s | %7.1f %7.1f %+7.1f | %7.0f %7.0f | %7.1f %7.1f | %-28s%s"
              % (c["name"], tl_str,
                 r_vmin, t_vmin, t_vmin - r_vmin,
                 r_brk_max, t_brk_max,
                 r_wot, t_wot,
                 diag, flag))

        if tl > 0.05:
            priority_list.append({
                "name": c["name"], "time_loss": tl, "diag": diag,
                "vmin_diff": t_vmin - r_vmin, "brk_diff": t_brk_max - r_brk_max,
                "wot_diff": t_wot - r_wot, "r_vmin": r_vmin, "t_vmin": t_vmin,
                "r_brk": r_brk_max, "t_brk": t_brk_max,
            })

    total_corner_loss = sum(p["time_loss"] for p in priority_list)
    print("  " + "=" * (W - 2))
    print("  Cumulative corner time loss: %+.3fs  |  Whole-lap delta: %+.3fs"
          % (total_corner_loss, total_delta))

    # ── Priority coaching output ─────────────────────────────────────────────
    if priority_list:
        priority_list.sort(key=lambda x: -x["time_loss"])
        print()
        print("  PRIORITY COACHING (fastest to slowest improvement opportunity)")
        print("  " + "-" * 80)
        for i, p in enumerate(priority_list, 1):
            coaching = _coaching_tip(p)
            print("  %d. %-22s  %+.3fs   %s" % (i, p["name"], p["time_loss"], p["diag"]))
            print("     Apex speed: %.1f → %.1f km/h (%+.1f)   Peak brake: %.0f → %.0f kPa   WOT: %+.1f%%"
                  % (p["r_vmin"], p["t_vmin"], p["vmin_diff"],
                     p["r_brk"], p["t_brk"], p["wot_diff"]))
            print("     → %s" % coaching)
            print()

    print("=" * W + "\n")
    return priority_list


def _coaching_tip(p):
    """Generate a specific coaching sentence from diagnostic data."""
    diag = p["diag"]
    dv   = p["vmin_diff"]
    dbk  = p["brk_diff"]
    dwot = p["wot_diff"]

    if "Insufficient Braking" in diag:
        return ("Brake later and harder into this corner. "
                "Ref brakes %.0f kPa; you used %.0f kPa. "
                "Trust the car — commit to a later, sharper brake point."
                % (p["r_brk"], p["t_brk"]))
    if "Low Apex" in diag:
        return ("Over-braking reduces apex speed. "
                "Try releasing the brake sooner (trail braking) "
                "to carry %.1f more km/h through the apex." % abs(dv))
    if "Early Braking" in diag:
        return ("Move brake point %.1f–2 car-lengths later. "
                "You're bleeding speed before it's needed." % abs(dv / 10))
    if "Late / Hesitant Throttle" in diag:
        return ("Pick up throttle earlier on exit. "
                "You're leaving %.1f%% WOT time on the table vs reference."
                % abs(dwot))
    return ("General pace deficit. Review speed trace overlay in MoTeC i2 "
            "for a detailed braking / throttle trace comparison.")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref_file",    nargs="?", help="Reference (benchmark) file (.rcz/.ld/.csv)")
    parser.add_argument("target_file", nargs="?", help="Target (comparison) file (.rcz/.ld/.csv)")
    parser.add_argument("--ref_lap",    metavar="MM:SS.mmm",
                        help="Pick a specific lap from the reference file by lap time")
    parser.add_argument("--target_lap", metavar="MM:SS.mmm",
                        help="Pick a specific lap from the target file by lap time")
    parser.add_argument("--dir",  metavar="DIR",
                        help="Directory of .ld files; fastest is benchmark, rest are targets")
    parser.add_argument("--track", default=DEFAULT_PRESET,
                        choices=list(CORNER_PRESETS.keys()),
                        help="Corner preset to use (default: %s)" % DEFAULT_PRESET)
    parser.add_argument("--tol", type=float, default=2.0,
                        help="Tolerance in seconds when matching --ref_lap/--target_lap (default 2.0)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    corners = CORNER_PRESETS[args.track]

    # ── Batch directory mode ─────────────────────────────────────────────────
    if args.dir:
        files = sorted([
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir) if f.endswith(".ld")
        ])
        if len(files) < 2:
            print("ERROR: Need at least 2 .ld files in '%s'." % args.dir)
            sys.exit(1)
        laps = []
        for f in files:
            r = load_lap(f, verbose=args.verbose)
            if r:
                laps.append(r)
        if len(laps) < 2:
            sys.exit(1)
        laps.sort(key=lambda x: x["lap_time_sec"])
        ref = laps[0]
        for tgt in laps[1:]:
            analyze_and_report(ref, tgt, corners, verbose=args.verbose)
        sys.exit(0)

    # ── Two-file mode ────────────────────────────────────────────────────────
    if not args.ref_file or not args.target_file:
        parser.print_help()
        sys.exit(1)

    ref_sec = parse_lap_time_str(args.ref_lap)
    tgt_sec = parse_lap_time_str(args.target_lap)

    print("Loading reference:  %s  (lap: %s)"
          % (args.ref_file, fmt_lap_time(ref_sec) if ref_sec else "fastest"))
    ref = load_lap(args.ref_file, ref_sec, args.tol, args.verbose)

    print("Loading target:     %s  (lap: %s)"
          % (args.target_file, fmt_lap_time(tgt_sec) if tgt_sec else "fastest"))
    tgt = load_lap(args.target_file, tgt_sec, args.tol, args.verbose)

    if not ref or not tgt:
        print("ERROR: Could not extract lap data. Use -v for details.")
        sys.exit(1)

    print("Reference lap : %s  (%.3fs)" % (fmt_lap_time(ref["lap_time_sec"]), ref["lap_time_sec"]))
    print("Target lap    : %s  (%.3fs)" % (fmt_lap_time(tgt["lap_time_sec"]), tgt["lap_time_sec"]))

    analyze_and_report(ref, tgt, corners, verbose=args.verbose)


if __name__ == "__main__":
    main()
