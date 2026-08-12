"""Vehicle dynamics & math channel derivation (extracted from DataLog methods)."""

from __future__ import annotations

import numpy as np

from core.models import Message
from constants import (
    CH_ACCELERATOR_POS,
    CH_BRAKE_POS,
    CH_BRAKE_PRESS,
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LAT_SMOOTH,
    CH_CG_ACCEL_LON,
    CH_CG_ACCEL_LON_SMOOTH,
    CH_GPS_HEADING,
    CH_GROUND_SPEED,
    CH_G_COMBINED,
    CH_SLIP_ANGLE_FL,
    CH_SLIP_ANGLE_FR,
    CH_SLIP_ANGLE_RL,
    CH_SLIP_ANGLE_RR,
    CH_STEERING_ANGLE,
    CH_THROTTLE_POS,
    CH_UNDERSTEER_INDEX,
    CH_YAW_RATE,
)


DEFAULT_GEAR_RATIO_THRESHOLDS = (110.0, 70.0, 52.0, 42.0, 33.0, 20.0)


def calculate_math_channels(data_log, g_source="auto", kinematics=False,
                            gear_ratio_thresholds=None):
    """
    g_source: "auto" (default), "sensor", or "calc"
      - "auto": Use raw IMU sensor G channels if present; otherwise derive from GPS.
      - "sensor": Only use raw IMU sensor G channels. Do not derive G from GPS.
      - "calc": Force deriving G channels from GPS Speed + Yaw Rate, overriding sensor channels.
    """
    derive_yaw_rate_from_gps_heading(data_log)

    if g_source == "calc":
        if CH_CG_ACCEL_LAT in data_log.channels:
            del data_log.channels[CH_CG_ACCEL_LAT]
        if CH_CG_ACCEL_LON in data_log.channels:
            del data_log.channels[CH_CG_ACCEL_LON]
        derive_cg_accel_lateral(data_log, force=True)
        derive_cg_accel_longitudinal(data_log, force=True)
    elif g_source == "sensor":
        pass
    else:  # "auto"
        derive_cg_accel_lateral(data_log, force=False)
        derive_cg_accel_longitudinal(data_log, force=False)

    derive_smoothed_accel(data_log)
    if kinematics:
        calculate_kinematics(data_log)
    calculate_g_sum(data_log)
    derive_brake_pos(data_log)
    calculate_input_rates(data_log)
    mirror_throttle_accel(data_log)
    derive_gear_from_rpm_speed(data_log, ratio_thresholds=gear_ratio_thresholds)


def derive_gear_from_rpm_speed(data_log, ratio_thresholds=None):
    """Derive integer Gear (1-6) from timestamp-aligned RPM/speed ratios."""
    from constants import CH_GEAR, CH_ENGINE_RPM, CH_GROUND_SPEED
    if CH_GEAR in data_log.channels:
        return
    if CH_ENGINE_RPM not in data_log.channels or CH_GROUND_SPEED not in data_log.channels:
        return
    rpm_chan = data_log.channels[CH_ENGINE_RPM]
    spd_chan = data_log.channels[CH_GROUND_SPEED]
    if len(rpm_chan.messages) < 2 or len(spd_chan.messages) < 2:
        return

    rpm_times = np.asarray(rpm_chan.timestamps, dtype=np.float64)
    rpm_values = np.asarray(rpm_chan.values, dtype=np.float64)
    order = np.argsort(rpm_times, kind="stable")
    rpm_times = rpm_times[order]
    rpm_values = rpm_values[order]
    keep = np.concatenate([rpm_times[1:] != rpm_times[:-1], [True]])
    rpm_times = rpm_times[keep]
    rpm_values = rpm_values[keep]
    if len(rpm_times) < 2:
        return

    speed_times = np.asarray(spd_chan.timestamps, dtype=np.float64)
    speed_values = np.asarray(spd_chan.values, dtype=np.float64)
    order = np.argsort(speed_times, kind="stable")
    speed_times = speed_times[order]
    speed_values = speed_values[order]
    keep = np.concatenate([speed_times[1:] != speed_times[:-1], [True]])
    speed_times = speed_times[keep]
    speed_values = speed_values[keep]
    overlap = (speed_times >= rpm_times[0]) & (speed_times <= rpm_times[-1])
    if np.count_nonzero(overlap) < 2:
        return

    t_arr = speed_times[overlap]
    spd_vals = speed_values[overlap].copy()
    rpm_vals = np.interp(t_arr, rpm_times, rpm_values)
    if spd_chan.units == "mph":
        spd_vals *= 1.60934
    elif spd_chan.units == "m/s":
        spd_vals *= 3.6

    thresholds = ratio_thresholds
    if thresholds is None:
        thresholds = data_log.metadata.get("gear_ratio_thresholds", DEFAULT_GEAR_RATIO_THRESHOLDS)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (6,) or not np.all(np.diff(thresholds) < 0):
        raise ValueError("gear ratio thresholds must contain six strictly descending values")

    ratios = np.zeros(len(t_arr), dtype=np.float64)
    mask = (spd_vals > 5.0) & (rpm_vals > 500.0)
    ratios[mask] = rpm_vals[mask] / spd_vals[mask]

    gear_vals = np.zeros(len(t_arr), dtype=np.float64)
    unassigned = mask.copy()
    for gear, threshold in enumerate(thresholds, start=1):
        selected = unassigned & (ratios > threshold)
        gear_vals[selected] = float(gear)
        unassigned[selected] = False

    data_log.add_channel(CH_GEAR, "", float, 0)
    data_log.channels[CH_GEAR].set_samples(t_arr, gear_vals)


def derive_smoothed_accel(data_log, window_sec=0.5):
    """ Derive 0.5s moving average smoothed G channels for clean G-G diagrams in MoTeC. """
    for raw_name, smooth_name in [(CH_CG_ACCEL_LAT, CH_CG_ACCEL_LAT_SMOOTH),
                                  (CH_CG_ACCEL_LON, CH_CG_ACCEL_LON_SMOOTH)]:
        if raw_name in data_log.channels and smooth_name not in data_log.channels:
            ch = data_log.channels[raw_name]
            if len(ch.messages) < 5:
                continue
            freq = ch.avg_frequency()
            w = max(1, int(window_sec * freq))
            vals = np.array([m.value for m in ch.messages], dtype=np.float64)

            padded = np.pad(vals, (w // 2, w - 1 - w // 2), mode="edge")
            windows = np.lib.stride_tricks.sliding_window_view(padded, w)
            smoothed = np.mean(windows, axis=1)[:len(vals)]

            data_log.add_channel(smooth_name, ch.units, float, ch.decimals)
            data_log.channels[smooth_name].messages = [
                Message(ch.messages[i].timestamp, float(smoothed[i])) for i in range(len(vals))
            ]


def derive_cg_accel_lateral(data_log, force=False):
    if not force and CH_CG_ACCEL_LAT in data_log.channels:
        return
    if CH_GROUND_SPEED not in data_log.channels or CH_YAW_RATE not in data_log.channels:
        return
    spd_chan = data_log.channels[CH_GROUND_SPEED]
    yaw_chan = data_log.channels[CH_YAW_RATE]
    n = len(spd_chan.messages)
    if n < 1:
        return
    t_arr = np.array([m.timestamp for m in spd_chan.messages])
    vx_ms = np.array([m.value / 3.6 if spd_chan.units == "km/h" else m.value for m in spd_chan.messages])
    yr_rad = np.radians([m.value for m in yaw_chan.messages])
    ay = (vx_ms * yr_rad) / 9.80665
    data_log.add_channel(CH_CG_ACCEL_LAT, "G", float, 2)
    data_log.channels[CH_CG_ACCEL_LAT].messages = [Message(t_arr[i], ay[i]) for i in range(n)]


def derive_cg_accel_longitudinal(data_log, force=False):
    if not force and CH_CG_ACCEL_LON in data_log.channels:
        return
    if CH_GROUND_SPEED not in data_log.channels:
        return
    spd_chan = data_log.channels[CH_GROUND_SPEED]
    n = len(spd_chan.messages)
    if n < 2:
        return
    t_arr = np.array([m.timestamp for m in spd_chan.messages])
    vx_ms = np.array([m.value / 3.6 if spd_chan.units == "km/h" else m.value for m in spd_chan.messages])
    ax = np.gradient(vx_ms, t_arr) / 9.80665
    data_log.add_channel(CH_CG_ACCEL_LON, "G", float, 2)
    data_log.channels[CH_CG_ACCEL_LON].messages = [Message(t_arr[i], ax[i]) for i in range(n)]


_KINEMATICS_STEERING_RATIO = 13.5
_KINEMATICS_WHEELBASE_M = 2.575
_KINEMATICS_CG_TO_FRONT_AXLE_M = 1.25
_KINEMATICS_CG_TO_REAR_AXLE_M = 1.325
_KINEMATICS_LAT_VEL_TAU_S = 2.0


def calculate_kinematics(data_log):
    required = [CH_GROUND_SPEED, CH_CG_ACCEL_LAT, CH_YAW_RATE]
    if not all(r in data_log.channels for r in required):
        return
    vx_chan = data_log.channels[CH_GROUND_SPEED]
    n = min(len(data_log.channels[r].messages) for r in required)
    if n < 2:
        return
    time = np.array([m.timestamp for m in vx_chan.messages[:n]])
    vx = np.array([m.value for m in vx_chan.messages[:n]])
    if vx_chan.units == "km/h":
        vx /= 3.6
    ay = np.array([m.value * 9.80665 for m in data_log.channels[CH_CG_ACCEL_LAT].messages[:n]])
    yaw_rate_degs = np.array([m.value for m in data_log.channels[CH_YAW_RATE].messages[:n]])
    yaw_rate = np.radians(yaw_rate_degs * -1.0)

    dt = np.zeros(n)
    dt[1:] = np.diff(time)

    vy = np.zeros(n)
    beta = np.zeros(n)
    tau = 2.0

    for i in range(1, n):
        vy_dot = ay[i] - (vx[i] * yaw_rate[i])
        alpha = np.exp(-dt[i] / tau)
        vy[i] = (vy[i - 1] + vy_dot * dt[i]) * alpha
        if abs(ay[i]) < 0.49 and abs(yaw_rate_degs[i]) < 1.0:
            vy[i] = 0.0
        if vx[i] > 5.0:
            beta[i] = np.arctan2(vy[i], vx[i])

    ratio = 13.5
    wheelbase = 2.575
    lf = 1.25
    lr = 1.325

    slip_f = np.zeros(n)
    slip_r = np.zeros(n)
    steer_rad = np.zeros(n)
    if CH_STEERING_ANGLE in data_log.channels:
        steer_deg = np.array([m.value for m in data_log.channels[CH_STEERING_ANGLE].messages])
        steer_rad = np.radians(steer_deg / ratio)

    for i in range(n):
        if vx[i] > 5.0:
            slip_f[i] = np.degrees(steer_rad[i] - np.arctan2(vy[i] + yaw_rate[i] * lf, vx[i]))
            slip_r[i] = np.degrees(-np.arctan2(vy[i] - yaw_rate[i] * lr, vx[i]))

    for name in (CH_SLIP_ANGLE_FL, CH_SLIP_ANGLE_FR, CH_SLIP_ANGLE_RL, CH_SLIP_ANGLE_RR):
        data_log.add_channel(name, "deg", float, 2)
        src_data = slip_f if "F" in name else slip_r
        data_log.channels[name].messages = [Message(time[i], src_data[i]) for i in range(n)]

    us_index = np.zeros(n)
    for i in range(n):
        if vx[i] > 5.0:
            us_index[i] = np.degrees(steer_rad[i]) - np.degrees(wheelbase * yaw_rate[i] / vx[i])
    data_log.add_channel(CH_UNDERSTEER_INDEX, "deg", float, 2)
    data_log.channels[CH_UNDERSTEER_INDEX].messages = [Message(time[i], us_index[i]) for i in range(n)]


def calculate_g_sum(data_log):
    if CH_CG_ACCEL_LON not in data_log.channels or CH_CG_ACCEL_LAT not in data_log.channels:
        return
    ax_msgs = data_log.channels[CH_CG_ACCEL_LON].messages
    ay_msgs = data_log.channels[CH_CG_ACCEL_LAT].messages
    n = min(len(ax_msgs), len(ay_msgs))
    if n < 2:
        return
    time_g = [ax_msgs[i].timestamp for i in range(n)]
    ax = np.array([ax_msgs[i].value for i in range(n)])
    ay_g = np.array([ay_msgs[i].value for i in range(n)])
    g_sum = np.sqrt(ax ** 2 + ay_g ** 2)
    data_log.add_channel(CH_G_COMBINED, "G", float, 2)
    data_log.channels[CH_G_COMBINED].messages = [Message(time_g[i], g_sum[i]) for i in range(n)]


def derive_brake_pos(data_log):
    if CH_BRAKE_PRESS not in data_log.channels or CH_BRAKE_POS in data_log.channels:
        return
    press_chan = data_log.channels[CH_BRAKE_PRESS]
    n = len(press_chan.messages)
    if n < 1:
        return
    time_p = [m.timestamp for m in press_chan.messages]
    press_vals = np.array([m.value for m in press_chan.messages])
    if press_chan.units == "bar":
        press_vals *= 100.0
    bpos = np.clip(press_vals / 96.0, 0.0, 100.0)
    data_log.add_channel(CH_BRAKE_POS, "%", float, 2)
    data_log.channels[CH_BRAKE_POS].messages = [Message(time_p[i], bpos[i]) for i in range(n)]


def calculate_input_rates(data_log):
    calculate_rate(data_log, CH_STEERING_ANGLE, "deg/s")
    calculate_rate(data_log, CH_THROTTLE_POS, "%/s")
    calculate_rate(data_log, CH_BRAKE_POS, "%/s")


def mirror_throttle_accel(data_log):
    if CH_THROTTLE_POS not in data_log.channels and CH_ACCELERATOR_POS in data_log.channels:
        src = data_log.channels[CH_ACCELERATOR_POS]
        data_log.add_channel(CH_THROTTLE_POS, "%", float, 2)
        data_log.channels[CH_THROTTLE_POS].messages = [Message(m.timestamp, m.value) for m in src.messages]


def calculate_rate(data_log, channel_name, unit):
    """ Helper to calculate the rate of change for a channel. """
    if channel_name in data_log.channels:
        chan = data_log.channels[channel_name]
        vals = np.array([m.value for m in chan.messages])
        times = np.array([m.timestamp for m in chan.messages])
        if len(times) < 2:
            return
        rate = np.zeros(len(times))
        dt = np.diff(times)
        dt[dt == 0] = 0.001  # Prevent div by zero
        rate[1:] = np.diff(vals) / dt

        new_name = channel_name + " Rate"
        data_log.add_channel(new_name, unit, float, 2)
        data_log.channels[new_name].messages = [Message(times[i], rate[i]) for i in range(len(times))]


def derive_yaw_rate_from_gps_heading(data_log):
    if CH_YAW_RATE in data_log.channels:
        return
    gps_h_chan = data_log.channels.get(CH_GPS_HEADING)
    if not gps_h_chan or len(gps_h_chan.messages) < 2:
        return
    times = np.array([m.timestamp for m in gps_h_chan.messages])
    headings = np.array([m.value for m in gps_h_chan.messages])
    h_unwrapped = np.unwrap(np.radians(headings))
    h_deg = np.degrees(h_unwrapped)
    yaw_rate_val = -np.gradient(h_deg, times)
    freq = gps_h_chan.avg_frequency()
    w = max(1, int(0.2 * freq))
    if len(yaw_rate_val) >= w and w > 1:
        padded = np.pad(yaw_rate_val, (w // 2, w - 1 - w // 2), mode="edge")
        yaw_rate_val = np.mean(np.lib.stride_tricks.sliding_window_view(padded, w), axis=1)[:len(yaw_rate_val)]
    spd_chan = data_log.channels.get(CH_GROUND_SPEED)
    if spd_chan and len(spd_chan.messages) == len(times):
        v_vals = np.array([m.value for m in spd_chan.messages])
        if spd_chan.units == "mph":
            v_vals *= 1.60934
        yaw_rate_val[v_vals < 5.0] = 0.0
    yaw_rate_val = np.clip(yaw_rate_val, -150.0, 150.0)
    data_log.add_channel(CH_YAW_RATE, "deg/s", float, 2)
    data_log.channels[CH_YAW_RATE].messages = [Message(times[i], yaw_rate_val[i]) for i in range(len(times))]
    data_log.metadata["yaw_rate_source"] = "gps_heading_derivative"


# Backward compatibility aliases
_derive_smoothed_accel = derive_smoothed_accel
_derive_cg_accel_lateral = derive_cg_accel_lateral
_derive_cg_accel_longitudinal = derive_cg_accel_longitudinal
_calculate_kinematics = calculate_kinematics
_calculate_g_sum = calculate_g_sum
_derive_brake_pos = derive_brake_pos
_calculate_input_rates = calculate_input_rates
_mirror_throttle_accel = mirror_throttle_accel
_calculate_rate = calculate_rate
_derive_yaw_rate_from_gps_heading = derive_yaw_rate_from_gps_heading
