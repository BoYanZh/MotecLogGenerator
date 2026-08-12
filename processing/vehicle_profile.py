"""Load and validate reusable vehicle dynamics profiles."""

from __future__ import annotations

import json
import math
import os


KINEMATICS_KEYS = {
    "steering_ratio",
    "wheelbase_m",
    "cg_to_front_axle_m",
    "cg_to_rear_axle_m",
    "lateral_velocity_tau_s",
}
PROFILE_KEYS = {
    "name",
    "vehicle_id",
    "vehicle_weight",
    "kinematics",
    "gear_ratio_thresholds",
}


def _positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"vehicle profile {label} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"vehicle profile {label} must be positive and finite")
    return value


def validate_vehicle_profile(profile):
    if not isinstance(profile, dict):
        raise ValueError("vehicle profile root must be a JSON object")
    unknown = set(profile) - PROFILE_KEYS
    if unknown:
        raise ValueError(f"unknown vehicle profile fields: {', '.join(sorted(unknown))}")

    result = dict(profile)
    kinematics = profile.get("kinematics")
    if kinematics is not None:
        if not isinstance(kinematics, dict):
            raise ValueError("vehicle profile kinematics must be a JSON object")
        missing = KINEMATICS_KEYS - set(kinematics)
        unknown = set(kinematics) - KINEMATICS_KEYS
        if missing:
            raise ValueError(
                f"vehicle profile kinematics missing fields: {', '.join(sorted(missing))}"
            )
        if unknown:
            raise ValueError(
                f"unknown vehicle profile kinematics fields: {', '.join(sorted(unknown))}"
            )
        validated = {
            key: _positive_number(kinematics[key], f"kinematics.{key}")
            for key in KINEMATICS_KEYS
        }
        axle_sum = validated["cg_to_front_axle_m"] + validated["cg_to_rear_axle_m"]
        if not math.isclose(axle_sum, validated["wheelbase_m"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "vehicle profile cg_to_front_axle_m + cg_to_rear_axle_m "
                "must equal wheelbase_m"
            )
        result["kinematics"] = validated

    thresholds = profile.get("gear_ratio_thresholds")
    if thresholds is not None:
        if not isinstance(thresholds, list) or len(thresholds) != 6:
            raise ValueError("vehicle profile gear_ratio_thresholds must contain six values")
        thresholds = [
            _positive_number(value, "gear_ratio_thresholds") for value in thresholds
        ]
        if any(a <= b for a, b in zip(thresholds, thresholds[1:])):
            raise ValueError("vehicle profile gear_ratio_thresholds must be strictly descending")
        result["gear_ratio_thresholds"] = thresholds

    if "vehicle_weight" in profile:
        result["vehicle_weight"] = _positive_number(
            profile["vehicle_weight"], "vehicle_weight"
        )
    return result


def load_vehicle_profile(path):
    profile_path = os.path.abspath(os.path.expanduser(path))
    try:
        with open(profile_path, encoding="utf-8") as profile_file:
            profile = json.load(profile_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid vehicle profile JSON: {exc}") from exc
    return validate_vehicle_profile(profile)
