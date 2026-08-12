#!/usr/bin/env python3
"""
12-Mesh-Block Joint Optimization alignment for CA-9 S (port of the
antigravity "STRICT 2D SPATIAL POSITION & HEADING ALIGNMENT" method).

Divides the sim track into 12 anchor blocks and optimizes a piecewise
CubicSpline rotation/translation field (theta/de/dn per anchor) so that
every segment of the sim trajectory aligns tightly (<5m) with the real
GPS trajectory.  Joint continuity is enforced by the clamped cubic
spline through the 12 anchors.

Original result (antigravity task-2122):
    0m-8300m  RMS 1.82m
    8300-end  RMS 2.01m  (tail previously ~18m)
    Overall   RMS 1.85m

Usage:
    python tools/align_ca9_mesh.py [--sim stint_7_raw.ld] [--real real.ld]
        [--t-start 1099.9] [--t-end 1550.3] [--dry-run]
"""

import argparse
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial import KDTree

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ldparser.ldparser import ldData
from processing.gps_utils import get_wgs84_geodesic_factors, enu_to_wgs84, wgs84_to_enu

ACTI_DIR = os.environ.get("ACTI_TELEM_DIR", "")
SIM_STINT7 = os.path.join(ACTI_DIR, 'highway_9_skidpad_&_ig_toyota_gr86_premium_&_stint_7.ld') if ACTI_DIR else ''
REAL_LD = r'data/canyon/session_20260802_153319_ca-9_s.ld'
MERGED_LD = r'data/canyon/session_ca-9_s_sim.ld'

ANCHORS = np.array([0.0, 1000.0, 2000.0, 3000.0, 4200.0, 5400.0,
                    6600.0, 7800.0, 8300.0, 8800.0, 9300.0, 0.0])  # last replaced by max_d


def load_sim_ac_coords(sim_path):
    """Load Car Coord X/Y (meters, AC local) from raw ACTI stint file."""
    ld = ldData.fromfile(sim_path)
    ch = {c.name.strip(): c for c in ld.channs}
    x = np.array(ch['Car Coord X'].data, dtype=float)
    y = np.array(ch['Car Coord Y'].data, dtype=float)
    spd = np.array(ch['Ground Speed'].data, dtype=float)
    freq = float(ch['Car Coord X'].freq)
    step = np.sqrt(np.diff(x, prepend=x[0]) ** 2 + np.diff(y, prepend=y[0]) ** 2)
    moving = (spd > 15.0) & (step < 20.0)
    return x[moving], y[moving], freq


def load_real_gps_window(real_path, t_start, t_end):
    """Load real GPS samples within [t_start, t_end] (moving only)."""
    ld = ldData.fromfile(real_path)
    ch = {c.name.strip(): c for c in ld.channs}
    lat = np.array(ch['GPS Latitude'].data, dtype=float)
    lon = np.array(ch['GPS Longitude'].data, dtype=float)
    spd = np.array(ch['Ground Speed'].data, dtype=float)
    freq = float(ch['GPS Latitude'].freq)
    t = np.arange(len(lat)) / freq
    mask = (t >= t_start) & (t <= t_end) & (spd > 15.0)
    return lat[mask], lon[mask], t[mask]


def resample_by_arc(points, n_out):
    """Resample a polyline to n_out points evenly spaced by arc length."""
    d = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    keep = np.concatenate([[True], d > 1e-9])
    p = points[keep]
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))])
    s_out = np.linspace(0.0, cum[-1], n_out)
    return np.column_stack([np.interp(s_out, cum, p[:, 0]), np.interp(s_out, cum, p[:, 1])])


def fit_12_mesh(sim_xy, real_pts, anchors, n_steps=2000, heading_weight=10.0,
                lock_start=True):
    """
    Optimize 12-anchor piecewise spline (theta/de/dn) aligning sim -> real.
    sim_xy: (n,2) AC coordinates relative to start.
    real_pts: (m,2) real ENU points.
    lock_start: if True, add a hard penalty so the fitted start lands exactly
                on the real start (0m start residual).
    Returns dict with transforms, errors and the evaluated fitted points.
    """
    dist = np.insert(np.cumsum(np.hypot(np.diff(sim_xy[:, 0]), np.diff(sim_xy[:, 1]))), 0, 0.0)
    max_d = dist[-1]
    anchors = anchors.copy()
    anchors[-1] = max_d
    steps_d = np.linspace(0.0, max_d, n_steps)
    x_a = np.interp(steps_d, dist, sim_xy[:, 0])
    y_a = np.interp(steps_d, dist, sim_xy[:, 1])

    tree = KDTree(real_pts)
    de_r = np.gradient(real_pts[:, 0])
    dn_r = np.gradient(real_pts[:, 1])
    heading_real = np.arctan2(de_r, dn_r)

    def eval_transform(params):
        cs_th = CubicSpline(anchors, params[0:12], bc_type='clamped')
        cs_de = CubicSpline(anchors, params[12:24], bc_type='clamped')
        cs_dn = CubicSpline(anchors, params[24:36], bc_type='clamped')
        ths = np.radians(cs_th(steps_d))
        des = cs_de(steps_d)
        dns = cs_dn(steps_d)
        sc = 1.0000
        ef = sc * (x_a * np.cos(ths) - y_a * np.sin(ths)) + des
        nf = sc * (x_a * np.sin(ths) + y_a * np.cos(ths)) + dns
        sim_pts = np.column_stack([ef, nf])
        dists, indices = tree.query(sim_pts)
        de_s = np.gradient(ef)
        dn_s = np.gradient(nf)
        heading_sim = np.arctan2(de_s, dn_s)
        hdg_diff = np.abs(np.arctan2(np.sin(heading_sim - heading_real[indices]),
                                     np.cos(heading_sim - heading_real[indices])))
        cost = np.mean(dists ** 2) + heading_weight * np.mean(hdg_diff ** 2)
        if lock_start:
            cost += 1000.0 * (np.hypot(ef[0] - real_pts[0, 0], nf[0] - real_pts[0, 1]) ** 2)
        return cost, dists, sim_pts, ths, des, dns, indices

    # Per-anchor local init (rotation+translation around each anchor)
    th_init, de_init, dn_init = [], [], []
    for anc in anchors:
        w_mask = np.abs(steps_d - anc) <= 350.0
        xa_w, ya_w = x_a[w_mask], y_a[w_mask]

        def cost_loc(p):
            de, dn, th_deg = p
            th = np.radians(th_deg)
            ef_w = (xa_w * np.cos(th) - ya_w * np.sin(th)) + de
            nf_w = (xa_w * np.sin(th) + ya_w * np.cos(th)) + dn
            pts_w = np.column_stack([ef_w, nf_w])
            dists_w, _ = tree.query(pts_w)
            return np.mean(dists_w ** 2)

        res = minimize(cost_loc, [0.0, 0.0, 180.0], method='Powell')
        de_init.append(res.x[0])
        dn_init.append(res.x[1])
        th_init.append(res.x[2])

    init_p = th_init + de_init + dn_init
    res_opt = minimize(lambda p: eval_transform(p)[0], init_p, method='Powell')
    cost_v, dists, sim_pts, ths, des, dns, indices = eval_transform(res_opt.x)

    return {
        'anchors': anchors, 'params': res_opt.x,
        'dists': dists, 'sim_pts': sim_pts, 'steps_d': steps_d,
        'x_a': x_a, 'y_a': y_a, 'cost': cost_v,
    }


def report(result, tail_start=8300.0):
    dists = result['dists']
    steps_d = result['steps_d']
    tail_mask = steps_d >= tail_start
    head_mask = ~tail_mask
    print('=== 12-MESH-BLOCK JOINT OPTIMIZATION (ported) ===')
    print('  0m ~ %gm Mean Spatial Error:  %.2f meters' % (tail_start, np.mean(dists[head_mask])))
    print('  0m ~ %gm RMS Spatial Error:   %.2f meters' % (tail_start, np.sqrt(np.mean(dists[head_mask] ** 2))))
    print()
    print('=== %gm ~ End TAIL SECTION ===' % tail_start)
    print('  Tail Mean Error: %.2f meters' % np.mean(dists[tail_mask]))
    print('  Tail RMS Error:  %.2f meters' % np.sqrt(np.mean(dists[tail_mask] ** 2)))
    print('  Tail Max Error:  %.2f meters' % np.max(dists[tail_mask]))
    print()
    print('=== OVERALL ===')
    print('  Overall RMS Error: %.2f meters' % np.sqrt(np.mean(dists ** 2)))
    print('  Overall Mean Error:%.2f meters' % np.mean(dists))
    print('  Overall Max Error: %.2f meters' % np.max(dists))
    return np.sqrt(np.mean(dists ** 2))


def transform_arc_points(sim_xy, fit, n_steps=None):
    """
    Apply the fitted 12-anchor spline transform to raw sim polyline points.
    Returns transformed ENU (n,2).
    """
    dist = np.insert(np.cumsum(np.hypot(np.diff(sim_xy[:, 0]), np.diff(sim_xy[:, 1]))), 0, 0.0)
    max_d = dist[-1]
    anchors = fit['anchors']
    params = fit['params']
    cs_th = CubicSpline(anchors, params[0:12], bc_type='clamped')
    cs_de = CubicSpline(anchors, params[12:24], bc_type='clamped')
    cs_dn = CubicSpline(anchors, params[24:36], bc_type='clamped')
    ths = np.radians(cs_th(dist))
    des = cs_de(dist)
    dns = cs_dn(dist)
    sc = 1.0000
    ef = sc * (sim_xy[:, 0] * np.cos(ths) - sim_xy[:, 1] * np.sin(ths)) + des
    nf = sc * (sim_xy[:, 0] * np.sin(ths) + sim_xy[:, 1] * np.cos(ths)) + dns
    return np.column_stack([ef, nf])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sim', default=SIM_STINT7, help='raw ACTI stint_7 .ld with Car Coord X/Y')
    ap.add_argument('--real', default=REAL_LD, help='real .ld')
    ap.add_argument('--t-start', type=float, default=1099.9, help='real window start (s)')
    ap.add_argument('--t-end', type=float, default=1550.3, help='real window end (s)')
    ap.add_argument('--dry-run', action='store_true', help='fit + report only, do not write .ld')
    ap.add_argument('--no-lock-start', action='store_true', help='disable hard start-point lock')
    args = ap.parse_args()

    # 1. Sim side: AC coords (moving frames only)
    x_s, y_s, _ = load_sim_ac_coords(args.sim)
    sim_rel = np.column_stack([x_s - x_s[0], y_s - y_s[0]])
    print(f"sim moving frames: {len(sim_rel)}  arc={np.hypot(np.diff(sim_rel[:,0]), np.diff(sim_rel[:,1])).sum():.0f}m")

    # 2. Real side: GPS window -> ENU (meters) around start point
    lat_r, lon_r, t_r = load_real_gps_window(args.real, args.t_start, args.t_end)
    lat0, lon0 = float(lat_r[0]), float(lon_r[0])
    e_r, n_r = wgs84_to_enu(lat_r, lon_r, lat0, lon0)
    real_pts = np.column_stack([e_r, n_r])
    print(f"real window frames: {len(real_pts)}  arc={np.hypot(np.diff(real_pts[:,0]), np.diff(real_pts[:,1])).sum():.0f}m")

    # 3. Fit 12-mesh joint optimization
    fit = fit_12_mesh(sim_rel, real_pts, ANCHORS, lock_start=not args.no_lock_start)
    overall_rms = report(fit)

    # 4. Apply to full sim polyline (moving frames) for verification of start/end
    fitted = transform_arc_points(sim_rel, fit)
    d_start = np.hypot(*(fitted[0] - real_pts[0]))
    d_end = np.hypot(*(fitted[-1] - real_pts[-1]))
    print()
    print(f"SIM start -> REAL start: {d_start:.2f}m")
    print(f"SIM end   -> REAL end:   {d_end:.2f}m")

    if args.dry_run:
        print("\n[dry-run] not writing files.")
        return

    # 5. Write transformed GPS back into merged sim .ld
    #    The merged file is stint_4 (multi-lap).  The user's 5:37.450 lap is the
    #    window [0.1s, 337.6s] of the merged file.  We transform that window with
    #    the stint_7-calibrated 12-anchor spline by mapping arc length fraction:
    #    cumulative arc of the window (stationary frames add 0 arc) mapped onto
    #    stint_7's arc [0, max].  Other laps (beyond the window) are left as-is
    #    (old CFG positions) to avoid extrapolation.
    stint4_path = os.path.join(ACTI_DIR, 'highway_9_skidpad_&_ig_toyota_gr86_premium_&_stint_4.ld')
    s4 = ldData.fromfile(stint4_path)
    s4ch = {c.name.strip(): c for c in s4.channs}
    ac_x = np.array(s4ch['Car Coord X'].data, dtype=float)
    ac_y = np.array(s4ch['Car Coord Y'].data, dtype=float)
    sp4 = np.array(s4ch['Ground Speed'].data, dtype=float)
    ac_rel = np.column_stack([ac_x - ac_x[0], ac_y - ac_y[0]])

    merged = ldData.fromfile(MERGED_LD)
    mch = {c.name.strip(): c for c in merged.channs}
    # force-load every channel BEFORE writing (writer reads lazily from same path)
    for c in merged.channs:
        _ = c.data
    ml = mch['GPS Latitude'].data
    mn = mch['GPS Longitude'].data
    mfreq = float(mch['GPS Latitude'].freq)

    if len(ac_rel) != len(ml):
        print(f"ERROR: stint_4 frames {len(ac_rel)} != merged frames {len(ml)}; aborting write")
        return

    m_per_deg_lat, m_per_deg_lon = get_wgs84_geodesic_factors(lat0)

    # window [0s, 337.6s] in merged file (from t=0 so no untransformed lead-in)
    mt = np.arange(len(ml)) / mfreq
    win = mt <= 337.6
    win_idx = np.where(win)[0]

    # cumulative arc along the window (moving frames only add distance)
    ac_win = ac_rel[win_idx]
    arc_win = np.insert(np.cumsum(np.hypot(np.diff(ac_win[:, 0]), np.diff(ac_win[:, 1]))), 0, 0.0)
    # map onto stint_7 arc by fraction
    sim_dist = np.insert(np.cumsum(np.hypot(np.diff(sim_rel[:, 0]), np.diff(sim_rel[:, 1]))), 0, 0.0)
    frac = arc_win / (arc_win[-1] + 1e-9)
    sim_at = np.column_stack([np.interp(frac, sim_dist / sim_dist[-1], sim_rel[:, 0]),
                              np.interp(frac, sim_dist / sim_dist[-1], sim_rel[:, 1])])
    fitted_enu = transform_arc_points(sim_at, fit)

    lat_new = ml.copy()
    lon_new = mn.copy()
    lat_new[win_idx] = lat0 + fitted_enu[:, 1] / m_per_deg_lat
    lon_new[win_idx] = lon0 + fitted_enu[:, 0] / m_per_deg_lon

    mch['GPS Latitude'].data[:] = lat_new
    mch['GPS Longitude'].data[:] = lon_new

    tmp = MERGED_LD + '.tmp'
    merged.write(tmp)
    v = ldData.fromfile(tmp)
    vc = {c.name.strip(): c for c in v.channs}
    print(f"\nverify temp file: n={len(vc['GPS Latitude'].data)}")
    if len(vc['GPS Latitude'].data) == len(ml):
        os.replace(tmp, MERGED_LD)
        print(f"OK: replaced {MERGED_LD}")
    else:
        print("ERROR: write failed; original file untouched")


if __name__ == '__main__':
    main()
