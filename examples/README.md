# MotecLogGenerator Examples

This directory contains anonymized, privacy-safe sample files for testing and demonstrating the MoTeC log generator tool across different logger formats.

All personal identifiers (e.g. driver names, specific vehicle IDs, private locations) in these sample files have been sanitized.

## CAN
Files:
* `can_sample.log`
* `sample_can_spec.dbc`

Usage:
```bash
python3 ../motec_log_generator.py can_sample.log CAN --dbc sample_can_spec.dbc
```

## Generic CSV
Files:
* `csv_sample.csv`

Usage:
```bash
python3 ../motec_log_generator.py csv_sample.csv CSV
```

## PB Buddy
Files:
* `pbbuddy_sample.csv`

Usage:
```bash
python3 ../motec_log_generator.py pbbuddy_sample.csv PBBUDDY
# Or auto-detect:
python3 ../motec_log_generator.py pbbuddy_sample.csv AUTO
```

## AiM Solo / RaceStudio
Files:
* `aim_solo_sample.csv`

Usage:
```bash
python3 ../motec_log_generator.py aim_solo_sample.csv AIM
# Or auto-detect:
python3 ../motec_log_generator.py aim_solo_sample.csv AUTO
```

## COBB Accessport
Files:
* `accessport_sample.csv`

Usage:
```bash
python3 ../motec_log_generator.py accessport_sample.csv ACCESSPORT
```

## RaceChrono
Files:
* `racechrono_sample.csv`

Usage:
```bash
python3 ../motec_log_generator.py racechrono_sample.csv RACECHRONO
```
