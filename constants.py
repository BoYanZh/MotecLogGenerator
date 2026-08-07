"""
Centralized Constants & Channel Name Mapping Definitions for MotecLogGenerator.

Provides a Single Source of Truth for:
  - Canonical MoTeC channel names and standard units.
  - Multi-source channel alias mappings (iRacing, RaceChrono, AiM Solo, PB Buddy, VBOX, Cobb, Generic CSVs).
"""

import numpy as np

# ----------------------------------------------------------------------------
# Canonical MoTeC Channel Names (Single Source of Truth)
# ----------------------------------------------------------------------------
CH_GROUND_SPEED     = "Ground Speed"
CH_CG_ACCEL_LAT     = "CG Accel Lateral"
CH_CG_ACCEL_LON     = "CG Accel Longitudinal"
CH_GPS_LATITUDE     = "GPS Latitude"
CH_GPS_LONGITUDE    = "GPS Longitude"
CH_GPS_HEADING      = "GPS Heading"
CH_GPS_ALTITUDE     = "GPS Altitude"
CH_GPS_SATS         = "GPS Satellites"
CH_GPS_FIX          = "GPS Fix"
CH_LAP_NUMBER       = "Lap Number"
CH_THROTTLE_POS     = "Throttle Pos"
CH_BRAKE_PRESS      = "Brake Press"
CH_BRAKE_POS        = "Brake Pos"
CH_ENGINE_RPM       = "Engine RPM"
CH_STEERING_ANGLE   = "Steering Angle"
CH_COOLANT_TEMP     = "Coolant Temp"
CH_ENGINE_OIL_TEMP  = "Engine Oil Temp"
CH_GEAR             = "Gear"
CH_YAW_RATE         = "Chassis Yaw Rate"
CH_SLIP_ANGLE_FL    = "Tire Slip Angle FL"
CH_SLIP_ANGLE_FR    = "Tire Slip Angle FR"
CH_SLIP_ANGLE_RL    = "Tire Slip Angle RL"
CH_SLIP_ANGLE_RR    = "Tire Slip Angle RR"
CH_UNDERSTEER_INDEX = "Understeer Index"
CH_G_COMBINED       = "G Force Combined"

# ----------------------------------------------------------------------------
# Channel Alias Mappings: Raw header string -> (canonical_name, unit, dec)
# ----------------------------------------------------------------------------
CHANNEL_ALIASES = {
    # Speed
    "Ground Speed":           (CH_GROUND_SPEED, "km/h", 2),
    "GPS Speed":              (CH_GROUND_SPEED, "km/h", 2),
    "Speed(km/h)":            (CH_GROUND_SPEED, "km/h", 2),
    "Speed":                  (CH_GROUND_SPEED, "km/h", 2),
    "velocity":               (CH_GROUND_SPEED, "km/h", 2),
    "velocity-calc":          (CH_GROUND_SPEED, "km/h", 2),
    "velocity kmh":           (CH_GROUND_SPEED, "km/h", 2),
    "Vehicle Speed":          ("Vehicle Speed", "km/h", 2),
    "velocity-canbus":        ("Vehicle Speed", "km/h", 2),

    # GPS Coordinates
    "GPS Latitude":           (CH_GPS_LATITUDE, "deg", 7),
    "GPS Longitude":          (CH_GPS_LONGITUDE, "deg", 7),
    "Latitude":               (CH_GPS_LATITUDE, "deg", 7),
    "Longitude":              (CH_GPS_LONGITUDE, "deg", 7),
    "lat":                    (CH_GPS_LATITUDE, "deg", 7),
    "long":                   (CH_GPS_LONGITUDE, "deg", 7),

    # Accelerations
    "CG Accel Lateral":       (CH_CG_ACCEL_LAT, "G", 4),
    "GPS_LatAcc":             (CH_CG_ACCEL_LAT, "G", 4),
    "LateralAcc":             (CH_CG_ACCEL_LAT, "G", 4),
    "lateral_acc":            (CH_CG_ACCEL_LAT, "G", 4),
    "latacc":                 (CH_CG_ACCEL_LAT, "G", 4),
    "latacc-calc":            (CH_CG_ACCEL_LAT, "G", 4),
    "lat accel g":            (CH_CG_ACCEL_LAT, "G", 4),
    "CG Accel Longitudinal":  (CH_CG_ACCEL_LON, "G", 4),
    "GPS_LongAcc":            (CH_CG_ACCEL_LON, "G", 4),
    "LineAcc":                (CH_CG_ACCEL_LON, "G", 4),
    "LongAcc":                (CH_CG_ACCEL_LON, "G", 4),
    "longitudinal_acc":       (CH_CG_ACCEL_LON, "G", 4),
    "longacc":                (CH_CG_ACCEL_LON, "G", 4),
    "longacc-calc":           (CH_CG_ACCEL_LON, "G", 4),
    "long accel g":           (CH_CG_ACCEL_LON, "G", 4),

    # GPS Meta
    "GPS Heading":            (CH_GPS_HEADING, "deg", 2),
    "GPS_Bearing":            (CH_GPS_HEADING, "deg", 2),
    "Bearing":                (CH_GPS_HEADING, "deg", 2),
    "heading":                (CH_GPS_HEADING, "deg", 2),
    "GPS Altitude":           (CH_GPS_ALTITUDE, "m", 2),
    "GPS_Height":             (CH_GPS_ALTITUDE, "m", 2),
    "height":                 (CH_GPS_ALTITUDE, "m", 2),
    "GPS Satellites":         (CH_GPS_SATS, "", 0),
    "sats":                   (CH_GPS_SATS, "", 0),
    "satellites":             (CH_GPS_SATS, "", 0),

    # Lap Info
    "Lap Number":             (CH_LAP_NUMBER, "", 0),
    "Lap":                    (CH_LAP_NUMBER, "", 0),
    "Lap Count":              (CH_LAP_NUMBER, "", 0),
    "lap_number":             (CH_LAP_NUMBER, "", 0),

    # Driver Inputs & ECU
    "Throttle Pos":           (CH_THROTTLE_POS, "%", 2),
    "Throttle Position":      (CH_THROTTLE_POS, "%", 2),
    "PPS":                    (CH_THROTTLE_POS, "%", 2),
    "accelerator_pos":        (CH_THROTTLE_POS, "%", 2),
    "accelerator_pos-canbus": (CH_THROTTLE_POS, "%", 2),
    "Brake Press":            (CH_BRAKE_PRESS, "kPa", 2),
    "Brake Pressure":         (CH_BRAKE_PRESS, "kPa", 2),
    "BrakePress":             (CH_BRAKE_PRESS, "bar", 2),
    "brake_pressure":         (CH_BRAKE_PRESS, "kPa", 2),
    "brake_pressure-canbus":  (CH_BRAKE_PRESS, "kPa", 2),
    "Brake Pos":              (CH_BRAKE_POS, "%", 2),
    "brake_pos":              (CH_BRAKE_POS, "%", 2),
    "brake_pos-canbus":       (CH_BRAKE_POS, "%", 2),
    "Engine RPM":             (CH_ENGINE_RPM, "rpm", 2),
    "RPM":                    (CH_ENGINE_RPM, "rpm", 2),
    "rpm-canbus":             (CH_ENGINE_RPM, "rpm", 2),
    "Steering Angle":         (CH_STEERING_ANGLE, "deg", 2),
    "SteerAngle":             (CH_STEERING_ANGLE, "deg", 2),
    "steering_angle":         (CH_STEERING_ANGLE, "deg", 2),
    "steering_angle-canbus":  (CH_STEERING_ANGLE, "deg", 2),
    "Coolant Temp":           (CH_COOLANT_TEMP, "C", 2),
    "coolant_temp":           (CH_COOLANT_TEMP, "C", 2),
    "coolant_temp-canbus":    (CH_COOLANT_TEMP, "C", 2),
    "Engine Oil Temp":        (CH_ENGINE_OIL_TEMP, "C", 2),
    "engine_oil_temp":        (CH_ENGINE_OIL_TEMP, "C", 2),
    "engine_oil_temp-canbus": (CH_ENGINE_OIL_TEMP, "C", 2),
    "Gear":                   (CH_GEAR, "", 0),
}

# ----------------------------------------------------------------------------
# iRacing .ibt native channel -> MoTeC canonical mapping
# Each tuple: (ibt_var_name, ch_name, units, decimals, convert_fn_or_None)
# ----------------------------------------------------------------------------
_G = 9.80665
IBT_CHANNEL_MAP = [
    ("Speed",              CH_GROUND_SPEED,     "km/h",    2, lambda x: x * 3.6),
    ("Lat",                CH_GPS_LATITUDE,     "deg",     7, None),
    ("Lon",                CH_GPS_LONGITUDE,    "deg",     7, None),
    ("Alt",                CH_GPS_ALTITUDE,     "m",       2, None),
    ("LatAccel",           CH_CG_ACCEL_LAT,     "G",       4, lambda x: x / _G),
    ("LongAccel",          CH_CG_ACCEL_LON,     "G",       4, lambda x: x / _G),
    ("YawRate",            CH_YAW_RATE,         "deg/s",   3, np.degrees),
    ("YawNorth",           CH_GPS_HEADING,      "deg",     2, np.degrees),
    ("SteeringWheelAngle", CH_STEERING_ANGLE,   "deg",     2, np.degrees),
    ("Throttle",           CH_THROTTLE_POS,     "%",       2, lambda x: x * 100.0),
    ("Brake",              CH_BRAKE_POS,        "%",       2, lambda x: x * 100.0),
    ("RPM",                CH_ENGINE_RPM,       "rpm",     0, None),
    ("Gear",               CH_GEAR,             "",        0, None),
    ("WaterTemp",          CH_COOLANT_TEMP,     "C",       2, None),
    ("OilTemp",            CH_ENGINE_OIL_TEMP,  "C",       2, None),
    ("OilPress",           "Engine Oil Press",  "kPa",     2, lambda x: x * 100.0),
    ("ManifoldPress",      "Manifold Press",    "kPa",     2, lambda x: x * 100.0),
    ("FuelLevel",          "Fuel Level",        "l",       2, None),
    ("Voltage",            "Battery Voltage",   "V",       2, None),
    ("TrackTemp",          "Track Temp",        "C",       2, None),
    ("LapDistPct",         "Lap Distance",      "%",       2, lambda x: x * 100.0),
]

IBT_WHEEL_SPEED_MAP = [
    ("LFspeed", "Wheel Speed FL"),
    ("RFspeed", "Wheel Speed FR"),
    ("LRspeed", "Wheel Speed RL"),
    ("RRspeed", "Wheel Speed RR"),
]

IBT_BRAKE_PRESS_MAP = [
    ("LFbrakeLinePress", "Brake Press FL"),
    ("RFbrakeLinePress", "Brake Press FR"),
    ("LRbrakeLinePress", "Brake Press RL"),
    ("RRbrakeLinePress", "Brake Press RR"),
]
