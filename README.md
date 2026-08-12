# MotecLogGenerator

A high-performance Python utility for generating MoTeC `.ld` / `.ldx` log files from external telemetry sources (iRacing `.ibt`, RaceChrono `.rcz`, AIM `.xrk`/`.xrz`, Garmin `.fit`, AiM Solo / RaceStudio, Racelogic VBOX, COBB Accessport, PB Buddy, CAN bus logs, and generic CSVs). 

Generated log files automatically include the MoTeC Pro Analysis magic flag (`0xc81a4`), allowing them to be opened natively in both **MoTeC i2 Standard** and **MoTeC i2 Pro** with all advanced math and Pro features unlocked.

---

## Key Features

* **Multi-Format Parsing**:
  * **iRacing Native `.ibt` Telemetry**: Direct 60Hz binary parser for iRacing telemetry files (no Mu Exporter needed). Auto-extracts driver, car, track metadata from YAML session info.
  * **Racelogic VBOX `.vbo` Logs**: Parses NMEA latitude/longitude, velocity, 10Hz/20Hz/100Hz IMU accelerometers, gyroscopes, and CAN bus channels.
  * **RaceChrono Native `.rcz` Archives**: Direct binary unzipping, multi-stint auto-splitting, microsecond time-drift correction, and 20Hz/25Hz GPS.
  * **AIM `.xrk` / `.xrz` Native Logs**: Direct binary parsing of AIM data logger files (via `libxrk`), including compressed `.xrz` archives, GPS/lap detection, and driver/vehicle/venue metadata.
  * **Garmin `.fit` Native Logs**: Decodes Garmin Catalyst / watch / Edge FIT files (via `fitparse`) with GPS, speed, heart rate, power, cadence, and lap boundaries.
  * **PB Buddy & Generic CSVs**: Auto-detects custom column names, speed units (`mph` $\to$ `km/h`), and heading angles.
  * **COBB Accessport CSVs**: Resamples ECU channels cleanly without zero-order hold gaps.
  * **AiM Solo / RaceStudio CSVs**: Auto-maps `PPS`, `SteerAngle`, `BrakePress`, `RPM`, `Gear`, temperatures, and lap beacon markers.
  * **Raw CAN Bus Logs**: Parses raw CAN `.log` files paired with a `.dbc` file.
* **Vehicle Dynamics & Math Channels**:
  * **G-Force Source Selection (`--g-source {auto,sensor,calc}`)**: Choose between hardware IMU sensors or GPS Kinematic derivation ($a_{\text{lat}} = v \cdot \omega$).
  * **0.2s Kinematic Low-Pass Filter**: Eliminates high-speed GPS derivative jitter while preserving 100% of vehicle dynamics.
  * **Stationary Speed Guard**: Guards low-speed/stationary GPS wander ($v < 5\text{ km/h} \implies 0\text{ deg/s}$).
  * **Advanced Math Channels**: Computes `Tire Slip Angle FL/FR/RL/RR`, `Understeer Index`, `G Force Combined`, `Chassis Yaw Rate`, and 0.5s moving average smoothed G channels.
  * **Reusable Vehicle Profiles**: Loads validated JSON steering ratio, wheelbase, CG position, filter, metadata, and gear-ratio settings so non-GR86 vehicles do not rely on the built-in defaults.
* **Safe Output Pipeline**:
  * Writes `.ld` and `.ldx` to temporary files, reads both back for binary/XML verification, then replaces the final files.
  * Refuses to overwrite existing outputs unless `--force` is supplied and never replaces the source input. CSV inputs use `_export.csv` for auxiliary CSV output.
* **Sim & Real-World Overlay Compatibility**:
  * Fully coordinate-aligned with Assetto Corsa ACTI simulator logs for side-by-side real vs. sim telemetry comparison in MoTeC i2 Pro.

---

## Python Version & Dependencies

* **Python Version**: Python 3.8 or higher (3.8 ~ 3.12+)
* **Dependencies**: `numpy` (required), `cantools` (only required for CAN bus log type), `libxrk` (only required for AIM `.xrk`/`.xrz` log type), `fitparse` (only required for Garmin `.fit` log type)

Install dependencies via pip:
```bash
pip install numpy
pip install cantools  # only needed for CAN bus log processing
pip install libxrk    # only needed for AIM XRK/XRZ log processing
pip install fitparse  # only needed for Garmin FIT log processing
```

For development and the complete test suite (including all optional parsers):
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

---

## Usage Examples

### 1. iRacing Native `.ibt` Telemetry (Recommended)
```bash
# Convert iRacing .ibt telemetry file directly (no Mu Exporter needed)
python motec_log_generator.py "session.ibt" AUTO
```
*Reads 60Hz native binary format. Auto-extracts metadata (driver, car, track, date) from YAML session info by pressing Alt+L in iRacing to record.*

### 2. RaceChrono `.rcz` Native Archives
```bash
# Convert RaceChrono native .rcz archive directly
python motec_log_generator.py /path/to/session.rcz RCZ
```
*Auto-detects multi-stint sessions (multiple resumes during a track day) and exports separate `.ld` and `.ldx` files for each stint.*

### 3. AiM Solo & RaceChrono CSV Logs
```bash
# Convert AiM Solo or RaceChrono CSV log
python motec_log_generator.py /path/to/aim_or_racechrono.csv RACECHRONO
```
*Automatically maps channels like `PPS` (Throttle), `SteerAngle`, `BrakePress`, `OilTemp`, `ECT` with SI unit conversions (`psi -> kPa`, `°F -> °C`, `mph -> km/h`), and extracts lap beacon timestamps.*

### 4. AIM XRK / XRZ Native Logs
```bash
# Convert AIM data logger .xrk / compressed .xrz file
python motec_log_generator.py /path/to/session.xrk XRK
python motec_log_generator.py /path/to/session.xrz AUTO
```
*Binary parser backed by [`libxrk`](https://github.com/m3rlin45/libxrk). Auto-extracts laps, driver/vehicle/venue metadata, and standard channels (GPS, RPM, speed, G-forces).*

### 5. Garmin `.fit` Native Logs
```bash
# Convert Garmin Catalyst / watch / Edge FIT file
python motec_log_generator.py /path/to/session.fit FIT
python motec_log_generator.py /path/to/session.fit AUTO
```
*Decoder backed by [`fitparse`](https://github.com/dtcooper/python-fitparse). Extracts GPS, speed, altitude, heart rate, power, cadence, and lap boundaries.*

### 6. Automatic Log Type Detection (`AUTO`)
```bash
# Auto-detect log format (IBT, RCZ, XRK/XRZ, FIT, PB Buddy, AiM, RaceChrono, Accessport, etc.)
python motec_log_generator.py /path/to/session.ibt AUTO
python motec_log_generator.py /path/to/session_export.csv AUTO
```

### 7. CAN Bus & Accessport Logs
```bash
python motec_log_generator.py /path/to/can_data.log CAN --dbc /path/to/car.dbc
python motec_log_generator.py /path/to/accessport.csv ACCESSPORT
```

### 8. CSV Data Export (RaceStudio / Excel)
```bash
# Export any parsed log as a CSV data table alongside the .ld/.ldx files
python motec_log_generator.py /path/to/session.rcz RCZ --csv
python motec_log_generator.py /path/to/session.xrk XRK --csv
python motec_log_generator.py /path/to/session.fit FIT --csv --csv-wallclock
```
*Writes all resampled channels as a tabular CSV with a header row of MoTeC channel names (Time, GPS Latitude/Longitude, Speed, Engine RPM, ...). Importable by AIM RaceStudio / RaceChrono RaceStudio CSV import wizards or Excel. Use `--csv-wallclock` to write Unix wall-clock timestamps instead of elapsed seconds.*

### 9. Vehicle-Specific Kinematics

```bash
python motec_log_generator.py /path/to/session.csv AUTO \
  --kinematics \
  --vehicle-profile vehicle_profiles/gr86.json
```

The profile controls `steering_ratio`, `wheelbase_m`, front/rear CG-to-axle
distance, lateral-velocity filter time constant, and optional gear-ratio
thresholds. `--gear-ratio-thresholds` takes precedence when both are supplied.
Without a profile, `--kinematics` retains the built-in GR86/BRZ parameters and
prints a warning. The included
[`vehicle_profiles/gr86.json`](vehicle_profiles/gr86.json) is the reference
profile; copy it when defining another vehicle.

### Output Replacement

Existing `.ld`, `.ldx`, and requested auxiliary outputs are preserved by
default. Pass `--force` only when replacement is intentional:

```bash
python motec_log_generator.py session.fit AUTO --output converted.ld --force
```

---

## Command Line Options

The current input choices are `CAN`, `CSV`, `ACCESSPORT`, `RACECHRONO`, `RCZ`,
`PBBUDDY`, `VBO`, `IBT`, `XRK`, `FIT`, and `AUTO`. Run the CLI for the complete,
non-duplicated option reference:

```bash
python motec_log_generator.py --help
```

Key safety and dynamics options are `--output`, `--force`, `--kinematics`,
`--vehicle-profile`, `--gear-ratio-thresholds`, `--frequency`, and
`--mask-interp-gaps`.

---

## Repository Structure

* **`motec_log_generator.py`**: Main CLI entry point.
* **`data_log.py`**: Telemetry engine for parsing, resampling, filtering, and calculating advanced vehicle dynamics.
* **`motec_log.py`**: MoTeC binary `.ld` header packer and `.ldx` XML beacon generator.
* **`core/output.py`**: Temporary write, binary/XML read-back verification, and recoverable final-file replacement.
* **`processing/vehicle_profile.py`**: JSON vehicle profile validation.
* **`vehicle_profiles/`**: Reusable vehicle dynamics profiles, including the GR86/BRZ example.
* **`ldparser/`**: Low-level MoTeC `.ld` binary file parser/writer module.
* **`can_utils/`**: CAN bus helper utilities (`list_can_ids.py`).
* **`tools/`**: Helper scripts & analysis tools (`verify_log.py`, `convert_iracing_mu.py`, `convert_acti_log.py`, `analyze_tire_grip.py`, `analyze_lap_comparison.py`, `analyze_corner_time_loss.py`). See [`tools/README.md`](tools/README.md).
* **`tests/`**: Unit test suite.
* **`examples/`**: Sample telemetry logs and quickstart datasets.

---

## Helper Scripts & Tools (`tools/`)

Standalone helper utilities are located in the [`tools/`](tools) directory:

1. **`python tools/verify_log.py <log.ld>`**: Validates MoTeC `.ld` header integrity, channel bounds, and `.ldx` XML beacon sorting.
2. **`python tools/convert_iracing_mu.py <input_mu.ld>`**: *(Legacy: prefer native `.ibt` parser)* Converts 356-channel heavy iRacing Mu `.ld` logs into lightweight (~3-4MB) standardized MoTeC `.ld`/`.ldx` logs with strict canonical channel names, `m/s` → `km/h` speed scaling, DMS GPS combination, and auto lap beacons.
3. **`python tools/convert_acti.py <acti_log.ld>`**: Converts and aligns Assetto Corsa ACTI simulator `.ld` logs with real-world WGS84 GPS coordinates across any track via ICP point-cloud calibration (`tools/acti_track_gps.json`).
4. **`python tools/analyze_tire_grip.py --dir data/exported`**: Analyzes sustained 1s G-force grip metrics across MoTeC sessions.
5. **`python tools/analyze_lap_comparison.py --dir data/exported`**: Compares lap times, sector splits, and generates a ranked leaderboard across sessions.
6. **`python tools/analyze_corner_time_loss.py <ref.ld> <target.ld>`**: Analyzes corner-by-corner time loss ($\Delta t$), apex minimum speeds ($V_{\min}$), and driver performance diagnostics.

See [`tools/README.md`](tools/README.md) for detailed documentation.

---

## Running Tests

```bash
# Run available tests; optional parser tests are reported as skipped when their dependency is absent
python tests/test_examples.py

# Install every parser dependency and run the complete CI-equivalent suite
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover all supported input families, resampling, math channels, vehicle
profiles, CSV export, verified `.ld`/`.ldx` CLI round trips, overwrite protection,
and cleanup when staged-output verification fails.

---

## License & Disclaimer
This work was produced for research and track-day telemetry analysis purposes. It should in no way be used to circumvent MoTeC's licensing requirements for their data loggers or i2 analysis software.

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details. The `ldparser/` submodule (GPL-3.0 reverse-engineered MoTeC `.ld` parser) and the rest of the codebase are distributed under the same license.

---

## Acknowledgments & Credits

* **[Timur's ft86 repo](https://github.com/timurrrr/ft86)**: Special thanks to Timur for reverse-engineering and documenting Toyota GR86 / Subaru BRZ CAN PIDs, as well as providing PB Buddy sample CSV logs for testing and validation.
* **[ACTI (Assetto Corsa Telemetry Interface)](https://www.overtake.gg/downloads/acti-assetto-corsa-telemetry-interface.3948/)**: For establishing standard MoTeC telemetry structures for Assetto Corsa simulator logs.
* **smashndash / Saurabh**: Special thanks for providing sample AiM Solo & RaceStudio telemetry log files for testing and validation.
