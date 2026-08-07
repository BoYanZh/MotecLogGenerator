# MoTeC Telemetry Generator Tools & Utilities

This directory contains standalone helper scripts and analysis utilities for validating, aligning, and inspecting generated MoTeC `.ld` / `.ldx` telemetry files.

---

## 1. Log Verification Tool (`verify_log.py`)

A built-in verification utility that validates generated binary `.ld` and companion XML `.ldx` files prior to opening in MoTeC i2.

### Usage
```bash
python tools/verify_log.py /path/to/generated_log.ld
```

### What `verify_log.py` Validates:
* **Binary Header Integrity**: Unpacks MoTeC `.ld` binary headers and verifies driver, vehicle, track venue, and datetime metadata.
* **Channel Data & Bounds**: Checks sample counts, detects `NaN`/`Inf` anomalies, and validates physical bounds (e.g. GPS Latitude in `[-90, 90]`, Longitude in `[-180, 180]`).
* **Advanced Math Channels**: Confirms presence of pre-calculated vehicle dynamics channels (`Tire Slip Angle FL/FR/RL/RR`, `Understeer Index`, `G Force Combined`).
* **XML Lap Beacon Alignment (`.ldx`)**: Parses companion `.ldx` files to verify lap beacon timestamp sorting, fastest lap times, and total lap counts.

---

## 2. Assetto Corsa ACTI GPS Alignment Tool (`align_acti_gps.py`)

Aligns simulator telemetry logs from **ACTI (Assetto Corsa Telemetry Interface)** with real-world WGS84 GPS coordinates across any track, correcting 3D track coordinate offsets and North orientation mismatch for side-by-side overlay comparison in MoTeC i2 Pro.

### Usage
```bash
# Align a single ACTI .ld log or an entire directory of logs
python tools/align_acti_gps.py /path/to/acti/logs --output_dir data/acti_aligned

# Specify a track profile explicitly
python tools/align_acti_gps.py /path/to/acti/logs --track thunderhill_east_bypass
```

### Auto-Calibration Mode (`--calibrate`)
Automatically computes rotation angle $\theta$ and translation offsets $(dx, dy)$ for **any new Assetto Corsa track mod** by matching an ACTI simulator log against a real-world GPS log via ICP point-cloud optimization, saving the calibrated profile to [`acti_tracks.json`](file:///C:/Users/boyanzh/Desktop/Programs/repos/MotecLogGenerator/tools/acti_tracks.json):

```bash
# Calibrate a new track profile (e.g. Laguna Seca or Sonoma)
python tools/align_acti_gps.py --calibrate real_world_session.ld acti_sim_session.ld --track_key laguna_seca --track_name "Laguna Seca"
```

### Features:
* **Modular Track Config Profiles (`acti_tracks.json`)**: Pre-configured profiles for `thunderhill_east_bypass`, `thunderhill_ccw`, `laguna_seca`, `sonoma_raceway`, etc.
* **One-Command Auto Calibration**: Calculates `ref_lat`, `ref_lon`, `theta_deg`, `dx_m`, and `dy_m` with $<1.2\text{m}$ trajectory precision via Powell/ICP optimization.
* **In-Place Metadata Preservation**: Retains 100% of original ACTI header metadata (date, time, vehicle model, session type, tire comments).
* **Exact Lap & Sector Markers**: Copies original ACTI `.ldx` lap times and sector markers without false 0:00 out laps.
* **Standard GPS Channels**: Populates `GPS Latitude` and `GPS Longitude` for universal MoTeC i2 workspace template compatibility.

---

## 3. Tire Grip Analysis Tool (`analyze_tire_grip.py`)

Compares lateral grip capabilities across exported MoTeC sessions:

### Usage
```bash
python tools/analyze_tire_grip.py --dir data/exported --min_spd 60
```

### Metrics Evaluated:
* **`RawMax`**: Peak instantaneous $|a_y|$ (includes bumps/spikes)
* **`SustMax`**: Max 1s-sustained $|a_y|$ (floor of 1s sliding window, spike-free)
* **`P99`**: 99th percentile of $|a_y|$ while moving
* **`BestSeg`**: Duration (s) of the longest continuous $>0.7\text{G}$ cornering segment
* **`SegMean`**: Mean $|a_y|$ within that longest segment

---

## 4. Lap Time & Telemetry Leaderboard Tool (`compare_laps.py`)

Inspects exported MoTeC `.ld` / `.ldx` files across a directory and generates a sorted **Lap Time Leaderboard** with lap time deltas, fastest lap highlights ⭐, and sector split breakdowns.

### Usage
```bash
# Analyze all exported .ld sessions and display leaderboard & lap breakdowns
python tools/compare_laps.py --dir data/exported

# Only display the leaderboard (fastest lap per session)
python tools/compare_laps.py --dir data/exported --fastest_only
```

### Metrics & Features:
* **Leaderboard Ranking**: Ranks sessions by fastest lap time with delta gap (s) relative to the overall session benchmark.
* **Sector Split Breakdown**: Parses `.ldx` sector beacons (`S1`, `S2`, `S3`) to show split time progressions.
* **Metadata Extraction**: Unpacks driver, vehicle, track venue, session date, and lap counts directly from MoTeC headers.

---

## 5. Corner-by-Corner Time Loss Analysis Tool (`analyze_corner_time_loss.py`)

Calculates spatial distance-based time delta trace $dt(s)$ between a Target Lap and a Benchmark Reference Lap, analyzing **corner-by-corner time loss**, apex minimum speed ($V_{\min}$) differences, and primary loss diagnostics (e.g. over-braking, low apex speed, or late throttle application).

### Usage
```bash
# Compare a target session against a benchmark reference session
python tools/analyze_corner_time_loss.py benchmark_ref.ld target_session.ld

# Batch compare all exported .ld sessions against the overall fastest lap
python tools/analyze_corner_time_loss.py --dir data/exported
```

### Metrics & Diagnostics Evaluated:
* **`Time Loss (s)`**: Net time gained or lost ($\Delta t$) within each specific corner segment.
* **`Apex Speed (V_min)`**: Minimum cornering speed comparison in km/h ($V_{\min\text{,target}} - V_{\min\text{,ref}}$).
* **`Primary Cause / Diagnostic`**: Diagnoses performance deficiencies per corner (`Low Apex Speed / Over-braking`, `Early Braking on Entry`, `Hesitant / Late Throttle Application`).
