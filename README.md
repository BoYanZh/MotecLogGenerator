# MotecLogGenerator

A high-performance Python utility for generating MoTeC `.ld` / `.ldx` log files from external telemetry sources (iRacing `.ibt`, RaceChrono `.rcz`, AIM `.xrk`/`.xrz`, AiM Solo / RaceStudio, Racelogic VBOX, COBB Accessport, PB Buddy, CAN bus logs, and generic CSVs).

Generated log files automatically include the MoTeC Pro Analysis magic flag (`0xc81a4`), allowing them to be opened natively in both **MoTeC i2 Standard** and **MoTeC i2 Pro** with all advanced math and Pro features unlocked.

---

## Key Features

* **Multi-Format Parsing**:
  * **iRacing Native `.ibt` Telemetry**: Direct 60Hz binary parser for iRacing telemetry files (no Mu Exporter needed). Auto-extracts driver, car, track metadata from YAML session info.
  * **Racelogic VBOX `.vbo` Logs**: Parses NMEA latitude/longitude, velocity, 10Hz/20Hz/100Hz IMU accelerometers, gyroscopes, and CAN bus channels.
  * **RaceChrono Native `.rcz` Archives**: Direct binary unzipping, multi-stint auto-splitting, microsecond time-drift correction, and 20Hz/25Hz GPS.
  * **AIM `.xrk` / `.xrz` Native Logs**: Direct binary parsing of AIM data logger files (via `libxrk`), including compressed `.xrz` archives, GPS/lap detection, and driver/vehicle/venue metadata.
  * **PB Buddy & Generic CSVs**: Auto-detects custom column names, speed units (`mph` $\to$ `km/h`), and heading angles.
  * **COBB Accessport CSVs**: Resamples ECU channels cleanly without zero-order hold gaps.
  * **AiM Solo / RaceStudio CSVs**: Auto-maps `PPS`, `SteerAngle`, `BrakePress`, `RPM`, `Gear`, temperatures, and lap beacon markers.
  * **Raw CAN Bus Logs**: Parses raw CAN `.log` files paired with a `.dbc` file.
* **Vehicle Dynamics & Math Channels**:
  * **G-Force Source Selection (`--g-source {auto,sensor,calc}`)**: Choose between hardware IMU sensors or GPS Kinematic derivation ($a_{\text{lat}} = v \cdot \omega$).
  * **0.2s Kinematic Low-Pass Filter**: Eliminates high-speed GPS derivative jitter while preserving 100% of vehicle dynamics.
  * **Stationary Speed Guard**: Guards low-speed/stationary GPS wander ($v < 5\text{ km/h} \implies 0\text{ deg/s}$).
  * **Advanced Math Channels**: Computes `Tire Slip Angle FL/FR/RL/RR`, `Understeer Index`, `G Force Combined`, `Chassis Yaw Rate`, and 0.5s moving average smoothed G channels.
* **Sim & Real-World Overlay Compatibility**:
  * Fully coordinate-aligned with Assetto Corsa ACTI simulator logs for side-by-side real vs. sim telemetry comparison in MoTeC i2 Pro.

---

## Python Version & Dependencies

* **Python Version**: Python 3.8 or higher (3.8 ~ 3.12+)
* **Dependencies**: `numpy` (required), `cantools` (only required for CAN bus log type), `libxrk` (only required for AIM `.xrk`/`.xrz` log type)

Install dependencies via pip:
```bash
pip install numpy
pip install cantools  # only needed for CAN bus log processing
pip install libxrk    # only needed for AIM XRK/XRZ log processing
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

### 5. Automatic Log Type Detection (`AUTO`)
```bash
# Auto-detect log format (IBT, RCZ, XRK/XRZ, PB Buddy, AiM, RaceChrono, Accessport, etc.)
python motec_log_generator.py /path/to/session.ibt AUTO
python motec_log_generator.py /path/to/session_export.csv AUTO
```

### 6. CAN Bus & Accessport Logs
```bash
python motec_log_generator.py /path/to/can_data.log CAN --dbc /path/to/car.dbc
python motec_log_generator.py /path/to/accessport.csv ACCESSPORT
```

---

## Command Line Options

```text
usage: motec_log_generator.py [-h] [--output OUTPUT] [--g-source {auto,sensor,calc}]
                              [--frequency FREQUENCY] [--min_lap_sec MIN_LAP_SEC]
                              [--lap LAP] [--stint STINT] [--driver DRIVER]
                              [--vehicle_id VEHICLE_ID] [--venue_name VENUE_NAME]
                              [--event_name EVENT_NAME]
                              log {CAN,CSV,ACCESSPORT,RACECHRONO,RCZ,PBBUDDY,IBT,AUTO}

Options:
  --output OUTPUT          Path for output .ld file (default: same directory as input log)
  --g-source MODE          G-force channel source: 'auto' (use IMU sensor if present, fallback to GPS calc),
                           'sensor' (only IMU sensor), or 'calc' (force derive from GPS) (default: auto)
  --frequency FREQUENCY    Fixed frequency to resample all channels at or 'auto' (default: auto)
  --gpx                    Generate GPX track file (default: false)
  --kml                    Generate KML Google Earth track file (default: false)
  --min_lap_sec MIN_LAP    Minimum valid lap duration in seconds to filter noise (default: 15.0s)
  --mask-interp-gaps       Mask sample interpolation gaps (>1s) with NaN instead of interpolating through them (default: false)
  --lap LAP                Specific lap number to export (e.g. 1, 15) or 'all' (default: all)
  --stint STINT            Specific RCZ stint to export or 'all' for auto-split (default: all)
  --driver DRIVER          Driver name metadata (auto-extracted from log if omitted)
  --vehicle_id VEHICLE_ID  Vehicle model metadata (auto-extracted from log if omitted)
  --venue_name VENUE_NAME  Track venue metadata (auto-extracted from log if omitted)
```

---

## Repository Structure

* **`motec_log_generator.py`**: Main CLI entry point.
* **`data_log.py`**: Telemetry engine for parsing, resampling, filtering, and calculating advanced vehicle dynamics.
* **`motec_log.py`**: MoTeC binary `.ld` header packer and `.ldx` XML beacon generator.
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
# Run unit test suite (no extra dependencies required)
python tests/test_examples.py

# Or with pytest (optional)
pip install pytest
pytest tests/
```

Tests cover all input types (CAN/CSV/ACCESSPORT/RACECHRONO/PBBUDDY/AIM), resampling, math channels, G-force filtering, discrete value handling, and FIR filter design.

---

## License & Disclaimer
This work was produced for research and track-day telemetry analysis purposes. It should in no way be used to circumvent MoTeC's licensing requirements for their data loggers or i2 analysis software.

---

## Acknowledgments & Credits

* **[Timur's ft86 repo](https://github.com/timurrrr/ft86)**: Special thanks to Timur for reverse-engineering and documenting Toyota GR86 / Subaru BRZ CAN PIDs, as well as providing PB Buddy sample CSV logs for testing and validation.
* **[ACTI (Assetto Corsa Telemetry Interface)](https://www.overtake.gg/downloads/acti-assetto-corsa-telemetry-interface.3948/)**: For establishing standard MoTeC telemetry structures for Assetto Corsa simulator logs.
* **smashndash / Saurabh**: Special thanks for providing sample AiM Solo & RaceStudio telemetry log files for testing and validation.
