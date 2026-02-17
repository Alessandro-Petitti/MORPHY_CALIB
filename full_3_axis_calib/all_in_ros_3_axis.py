#!/usr/bin/env python3
# Procedura acquisizione consigliata (bag calibrazione 3 assi):
# 1) 3-5 s fermo iniziale
# 2) 8-12 s solo azimuth (destra/sinistra), poi 2-3 s pausa
# 3) 8-12 s solo elevation (su/giu), poi 2-3 s pausa
# 4) 8-12 s solo twist (torsione), poi 3-5 s fermo finale
# 5) Eseguire questo script per generare data/lut_3axis/*.csv
# 6) Eseguire full_3_axis_calib/lut.py per fit e verifica R^2
import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import rosbag

# =========================
# USER SETTINGS (edit these)
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BAG_ROOT_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, "full_3_axis_bags"))
BAG_PATH = BAG_ROOT_DIR
BAG_PATTERN = "full_3_axis_calib_*.bag"
ARM_INDEX = None  # None => infer from file name calib_arm_<idx>.bag
ROTOR_TOPIC = None  # Auto-detect if None (expects one /mocap/arm*/pose in bag)

# Fixed topics
MAG_TOPIC = "/mavros/mag_mux/raw"
JOINT_TOPIC = "/mocap/body/pose"

# Behavior / output
SHOW_PLOTS = True
INVERT_YZ = True
UNWRAP_AZIMUTH = True
# Keep wrapped by default: unwrapping can accumulate multi-turn offsets that are
# confusing during Hall-sensor calibration diagnostics.
UNWRAP_TWIST = False
ROTOR_Z_OFFSET = 6.499424548581256 / 1000
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "lut_3axis")
TWIST_CONDITION_WARN_MIN_NORM = 0.02
TWIST_REFERENCE_WINDOW_S = 2.0
TWIST_RECALIB_LOW_CONDITION_PCT = 0.5

# Automatic segmentation settings
SPEED_PCTILE_AZ = 75.0
SPEED_PCTILE_EL = 75.0
SPEED_PCTILE_TW = 60.0
DOMINANCE_RATIO_MIN_AZ = 1.35
DOMINANCE_RATIO_MIN_EL = 1.35
DOMINANCE_RATIO_MIN_TW = 1.10
MIN_SEGMENT_LEN_AZ = 40
MIN_SEGMENT_LEN_EL = 40
MIN_SEGMENT_LEN_TW = 8
SPEED_SMOOTH_WINDOW = 21
# Protocol-aware segmentation is the default for the guided acquisition flow
# documented at the top of this file.
PROTOCOL_PHASE_MODE = "force"  # "off", "auto", "force"
PROTOCOL_ACTIVE_PCTILE = 55.0
PROTOCOL_AXIS_PCTILE = 60.0
PROTOCOL_MIN_ACTIVE_LEN = 120

EPS = 1e-9
ARM_NAME_RE = re.compile(r"(?:^|_)arm_(\d+)\.bag$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build 3-axis LUT CSV from ROS bags. "
            "Input can be one .bag file or a directory with full_3_axis_calib_*.bag."
        )
    )
    parser.add_argument(
        "bag_input",
        nargs="?",
        default=BAG_PATH,
        help=(
            "Bag file path or directory. "
            "If directory, all files matching --bag-pattern are processed."
        ),
    )
    parser.add_argument(
        "--bag-pattern",
        default=BAG_PATTERN,
        help="Glob pattern used when bag_input is a directory.",
    )
    parser.add_argument(
        "--arm-index",
        type=int,
        default=None,
        help=(
            "Override mag_mux index for all bags. "
            "If omitted, uses ARM_INDEX setting or infers from file name."
        ),
    )
    return parser.parse_args()


def collect_bag_paths(bag_input, bag_pattern):
    bag_input = os.path.abspath(os.path.expanduser(bag_input))
    assert_in_bag_scope(bag_input)
    if os.path.isfile(bag_input):
        return [bag_input]
    if os.path.isdir(bag_input):
        pattern = os.path.join(bag_input, bag_pattern)
        bag_paths = sorted(glob.glob(pattern))
        files = [p for p in bag_paths if os.path.isfile(p)]
        for path in files:
            assert_in_bag_scope(path)
        return files
    raise FileNotFoundError(f"Input not found: {bag_input}")


def infer_arm_index_from_filename(bag_path):
    name = os.path.basename(bag_path)
    match = ARM_NAME_RE.search(name)
    if not match:
        raise RuntimeError(
            "Cannot infer ARM_INDEX from file name. Expected *arm_<idx>.bag, "
            f"got: {name}. Set --arm-index explicitly."
        )
    return int(match.group(1))


def assert_in_bag_scope(path):
    real_path = os.path.realpath(path)
    try:
        common = os.path.commonpath([real_path, BAG_ROOT_DIR])
    except ValueError:
        common = ""
    if common != BAG_ROOT_DIR:
        raise RuntimeError(
            f"Refusing bag outside 3-axis scope: {real_path}. Expected under {BAG_ROOT_DIR}"
        )


def unwrap_deg(angle_deg):
    return np.rad2deg(np.unwrap(np.deg2rad(angle_deg)))


def wrap_deg(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def transform_point(x, y, z, z_offset=0.0):
    z_value = z - z_offset
    if INVERT_YZ:
        return [x, -y, -z_value]
    return [x, y, z_value]


def interpolate_xyz(source_time, source_xyz, target_time):
    if len(source_time) < 2:
        raise RuntimeError("Not enough samples to interpolate.")
    out = np.zeros((len(target_time), source_xyz.shape[1]))
    for i in range(source_xyz.shape[1]):
        out[:, i] = np.interp(target_time, source_time, source_xyz[:, i])
    return out


def interpolate_nearest(source_time, source_values, target_time):
    if len(source_time) == 0:
        raise RuntimeError("Cannot interpolate nearest on empty source.")
    if len(source_time) == 1:
        return np.repeat(source_values[:1], len(target_time), axis=0)

    idx = np.searchsorted(source_time, target_time)
    idx = np.clip(idx, 1, len(source_time) - 1)
    left = idx - 1
    right = idx
    pick_right = (target_time - source_time[left]) > (source_time[right] - target_time)
    nearest_idx = np.where(pick_right, right, left)
    return source_values[nearest_idx]


def resolve_topic_name(bag, requested_topic):
    topics = set(bag.get_type_and_topic_info().topics.keys())
    if requested_topic in topics:
        return requested_topic

    available = ", ".join(sorted(topics))
    raise RuntimeError(
        f"Topic not found: {requested_topic}. Available topics: {available}"
    )


def detect_single_rotor_topic(bag):
    topics = sorted(bag.get_type_and_topic_info().topics.keys())
    rotor_candidates = []
    for t in topics:
        t_norm = t if t.startswith("/") else "/" + t
        if t_norm.startswith("/mocap/arm") and t_norm.endswith("/pose"):
            rotor_candidates.append(t)

    if len(rotor_candidates) == 1:
        return rotor_candidates[0]
    if not rotor_candidates:
        raise RuntimeError(
            "No rotor topic found. Expected one /mocap/arm*/pose topic in bag."
        )
    raise RuntimeError(
        "Multiple rotor topics found. Set ROTOR_TOPIC manually. "
        f"Candidates: {', '.join(rotor_candidates)}"
    )


def normalize_vectors(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, EPS)
    return v / safe_norms


def smooth_series(x, window):
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")


def normalize_quaternions(q):
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms < EPS):
        raise RuntimeError("Found near-zero quaternion norm.")
    return q / norms


def enforce_quaternion_continuity(q):
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i - 1], out[i]) < 0.0:
            out[i] *= -1.0
    return out


def average_quaternion(q):
    accum = np.zeros((4, 4), dtype=float)
    for qi in q:
        accum += np.outer(qi, qi)
    eigvals, eigvecs = np.linalg.eigh(accum)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    if q_avg[3] < 0.0:
        q_avg *= -1.0
    return q_avg


def quat_conjugate(q):
    out = q.copy()
    out[:, :3] *= -1.0
    return out


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.column_stack([x, y, z, w])


def quat_rotate_vectors(q, v):
    v_quat = np.column_stack([v, np.zeros(len(v))])
    return quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))[:, :3]


def calc_azimuth_elevation(joint_xyz, rotor_xyz):
    delta = rotor_xyz - joint_xyz
    x, y, z = delta[:, 0], delta[:, 1], delta[:, 2]
    azimuth = np.rad2deg(np.arctan2(y, x))
    elevation = np.rad2deg(np.arctan2(z, np.sqrt(x * x + y * y)))
    return azimuth, elevation


def extract_twist_from_relative(joint_quat, q_rel, joint_pos_raw, rotor_pos_raw):
    axis_world = normalize_vectors(rotor_pos_raw - joint_pos_raw)
    axis_joint = quat_rotate_vectors(quat_conjugate(joint_quat), axis_world)
    axis_joint = normalize_vectors(axis_joint)

    proj = np.sum(q_rel[:, :3] * axis_joint, axis=1, keepdims=True) * axis_joint
    q_twist_pre = np.column_stack([proj, q_rel[:, 3]])
    q_twist_norm = np.linalg.norm(q_twist_pre, axis=1)
    q_twist = q_twist_pre / np.maximum(q_twist_norm[:, None], EPS)
    # Use one quaternion hemisphere for principal-angle extraction.
    flip = q_twist[:, 3] < 0.0
    q_twist[flip] *= -1.0

    signed_sin_half = np.sum(q_twist[:, :3] * axis_joint, axis=1)
    twist_rad = 2.0 * np.arctan2(signed_sin_half, q_twist[:, 3])
    return np.rad2deg(twist_rad), q_twist_norm


def calc_twist_deg(
    time_s,
    joint_quat,
    rotor_quat,
    joint_pos_raw,
    rotor_pos_raw,
    return_conditioning=False,
):
    joint_quat = normalize_quaternions(joint_quat)
    rotor_quat = normalize_quaternions(rotor_quat)
    q_rel = quat_multiply(quat_conjugate(joint_quat), rotor_quat)
    q_rel = normalize_quaternions(q_rel)

    twist_base, cond_base = extract_twist_from_relative(
        joint_quat, q_rel, joint_pos_raw, rotor_pos_raw
    )
    low_condition_pct = 100.0 * float(np.mean(cond_base < TWIST_CONDITION_WARN_MIN_NORM))
    use_recalib = low_condition_pct >= TWIST_RECALIB_LOW_CONDITION_PCT

    if use_recalib:
        q_rel_cont = enforce_quaternion_continuity(q_rel)
        # Remove static mocap frame mounting offset using the initial stationary window.
        if len(time_s) > 0:
            t0 = time_s[0]
            ref_mask = time_s <= (t0 + TWIST_REFERENCE_WINDOW_S)
        else:
            ref_mask = np.zeros(0, dtype=bool)
        if np.sum(ref_mask) < 5:
            ref_count = min(len(q_rel_cont), 200)
            ref_mask = np.zeros(len(q_rel_cont), dtype=bool)
            ref_mask[:ref_count] = True

        q_ref = average_quaternion(q_rel_cont[ref_mask])[None, :]
        q_ref_inv = np.repeat(quat_conjugate(q_ref), len(q_rel_cont), axis=0)
        q_rel_corr = quat_multiply(q_ref_inv, q_rel_cont)
        q_rel_corr = normalize_quaternions(q_rel_corr)
        twist_deg, q_twist_norm = extract_twist_from_relative(
            joint_quat, q_rel_corr, joint_pos_raw, rotor_pos_raw
        )
    else:
        twist_deg, q_twist_norm = twist_base, cond_base

    if return_conditioning:
        return twist_deg, q_twist_norm, use_recalib, low_condition_pct
    return twist_deg


def prune_short_segments(mask, min_len):
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        start = i
        while i < n and out[i]:
            i += 1
        end = i
        if end - start < min_len:
            out[start:end] = False
    return out


def build_protocol_phase_masks(total_speed):
    n = len(total_speed)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

    active_thr = np.percentile(total_speed, PROTOCOL_ACTIVE_PCTILE)
    active = total_speed >= active_thr
    active = prune_short_segments(active, PROTOCOL_MIN_ACTIVE_LEN)

    if np.any(active):
        start = int(np.argmax(active))
        end = int(n - 1 - np.argmax(active[::-1]))
    else:
        start, end = 0, n - 1

    span = end - start + 1
    cut1 = start + span // 3
    cut2 = start + (2 * span) // 3
    cut1 = int(np.clip(cut1, start + 1, end))
    cut2 = int(np.clip(cut2, cut1 + 1, end + 1))

    phase_az = np.zeros(n, dtype=bool)
    phase_el = np.zeros(n, dtype=bool)
    phase_tw = np.zeros(n, dtype=bool)
    phase_az[start:cut1] = True
    phase_el[cut1:cut2] = True
    phase_tw[cut2 : end + 1] = True
    return phase_az, phase_el, phase_tw


def protocol_refine_axis_mask(base_mask, phase_mask, speed, min_len):
    if not np.any(phase_mask):
        return np.zeros_like(base_mask)
    phase_speed = speed[phase_mask]
    thr = np.percentile(phase_speed, PROTOCOL_AXIS_PCTILE)
    candidate = phase_mask & (speed >= thr)
    refined = (base_mask & phase_mask) | candidate
    refined = prune_short_segments(refined, min_len)
    return refined


def detect_axis_masks(time_s, azimuth, elevation, twist):
    def circular_speed_deg_s(angle_deg):
        if len(angle_deg) < 2:
            return np.zeros_like(angle_deg, dtype=float)
        dt = np.diff(time_s)
        safe_dt = np.maximum(dt, EPS)
        dtheta = wrap_deg(np.diff(angle_deg))
        speed = np.empty(len(angle_deg), dtype=float)
        speed[1:] = np.abs(dtheta) / safe_dt
        speed[0] = speed[1]
        return speed

    daz = circular_speed_deg_s(azimuth)
    delv = np.abs(np.gradient(elevation, time_s))
    dtw = circular_speed_deg_s(twist)

    daz = smooth_series(daz, SPEED_SMOOTH_WINDOW)
    delv = smooth_series(delv, SPEED_SMOOTH_WINDOW)
    dtw = smooth_series(dtw, SPEED_SMOOTH_WINDOW)
    speeds = np.column_stack([daz, delv, dtw])

    thresholds = np.array(
        [
            np.percentile(daz, SPEED_PCTILE_AZ),
            np.percentile(delv, SPEED_PCTILE_EL),
            np.percentile(dtw, SPEED_PCTILE_TW),
        ]
    )
    thresholds = np.maximum(thresholds, EPS)

    dominant_idx = np.argmax(speeds, axis=1)
    dominant_val = speeds[np.arange(len(speeds)), dominant_idx]
    sorted_vals = np.sort(speeds, axis=1)
    second_val = sorted_vals[:, 1]
    ratio = dominant_val / (second_val + EPS)

    ratio_min = np.array(
        [DOMINANCE_RATIO_MIN_AZ, DOMINANCE_RATIO_MIN_EL, DOMINANCE_RATIO_MIN_TW]
    )
    active = (dominant_val >= thresholds[dominant_idx]) & (
        ratio >= ratio_min[dominant_idx]
    )

    mask_az = prune_short_segments(active & (dominant_idx == 0), MIN_SEGMENT_LEN_AZ)
    mask_el = prune_short_segments(active & (dominant_idx == 1), MIN_SEGMENT_LEN_EL)
    mask_tw = prune_short_segments(active & (dominant_idx == 2), MIN_SEGMENT_LEN_TW)

    protocol_ready = (
        int(mask_az.sum()) >= MIN_SEGMENT_LEN_AZ
        and int(mask_el.sum()) >= MIN_SEGMENT_LEN_EL
        and int(mask_tw.sum()) >= MIN_SEGMENT_LEN_TW
    )
    if PROTOCOL_PHASE_MODE not in {"off", "auto", "force"}:
        raise ValueError(f"Unsupported PROTOCOL_PHASE_MODE: {PROTOCOL_PHASE_MODE}")
    apply_protocol = PROTOCOL_PHASE_MODE == "force" or (
        PROTOCOL_PHASE_MODE == "auto" and protocol_ready
    )
    if apply_protocol:
        phase_az, phase_el, phase_tw = build_protocol_phase_masks(daz + delv + dtw)
        mask_az = protocol_refine_axis_mask(mask_az, phase_az, daz, MIN_SEGMENT_LEN_AZ)
        mask_el = protocol_refine_axis_mask(mask_el, phase_el, delv, MIN_SEGMENT_LEN_EL)
        mask_tw = protocol_refine_axis_mask(mask_tw, phase_tw, dtw, MIN_SEGMENT_LEN_TW)

    return mask_az, mask_el, mask_tw, thresholds


def read_mag_data(bag, mag_topic, arm_index):
    mag_time = []
    mag_xyz = []
    lengths_seen = set()

    for _, msg, _ in bag.read_messages(topics=[mag_topic]):
        lengths_seen.add(len(msg.mags))
        if len(msg.mags) <= arm_index:
            continue
        mag = msg.mags[arm_index]
        mag_time.append(msg.header.stamp.to_sec())
        mag_xyz.append([mag.x, mag.y, mag.z])

    if not mag_time:
        raise RuntimeError(
            f"No valid samples on {mag_topic} for mag_mux[{arm_index}]. "
            f"Observed mags lengths: {sorted(lengths_seen)}"
        )
    return np.asarray(mag_time), np.asarray(mag_xyz)


def read_pose_topic(bag, topic, z_offset):
    pose_time = []
    pos_raw = []
    pos_transformed = []
    quat_xyzw = []

    for _, msg, _ in bag.read_messages(topics=[topic]):
        p = msg.pose.position
        q = msg.pose.orientation
        pose_time.append(msg.header.stamp.to_sec())

        raw_xyz = [p.x, p.y, p.z - z_offset]
        pos_raw.append(raw_xyz)
        pos_transformed.append(transform_point(p.x, p.y, p.z, z_offset=z_offset))
        quat_xyzw.append([q.x, q.y, q.z, q.w])

    if not pose_time:
        raise RuntimeError(f"No pose samples found on {topic}.")

    return (
        np.asarray(pose_time),
        np.asarray(pos_raw),
        np.asarray(pos_transformed),
        normalize_quaternions(np.asarray(quat_xyzw)),
    )


def read_mocap_data(bag, joint_topic, rotor_topic):
    j_time, j_raw, j_xyz, j_quat = read_pose_topic(bag, joint_topic, z_offset=0.0)
    r_time, r_raw, r_xyz, r_quat = read_pose_topic(
        bag,
        rotor_topic,
        z_offset=ROTOR_Z_OFFSET,
    )

    joint_raw_interp = interpolate_xyz(j_time, j_raw, r_time)
    joint_xyz_interp = interpolate_xyz(j_time, j_xyz, r_time)
    joint_quat_interp = interpolate_nearest(j_time, j_quat, r_time)
    joint_quat_interp = normalize_quaternions(joint_quat_interp)

    return r_time, joint_raw_interp, joint_xyz_interp, joint_quat_interp, r_raw, r_xyz, r_quat


def contiguous_spans(time_s, mask):
    spans = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        end = i - 1
        spans.append((time_s[start], time_s[end]))
    return spans


def add_mask_shading(ax, time_s, mask, color):
    for t0, t1 in contiguous_spans(time_s, mask):
        ax.axvspan(t0, t1, color=color, alpha=0.12, linewidth=0.0)


def build_output_path(bag_path, arm_index):
    bag_stem = os.path.splitext(os.path.basename(bag_path))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"{bag_stem}_arm{arm_index}.csv")


def plot_diagnostics(
    time_s,
    mag_xyz,
    azimuth,
    elevation,
    twist,
    mask_az,
    mask_el,
    mask_tw,
    bag_label,
):
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15, 10))
    if hasattr(fig.canvas, "manager") and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f"Preprocessing - {bag_label}")

    rows = [
        (axs[0], azimuth, mag_xyz[:, 0], "Azimuth [deg]", "mag x", mask_az, "#1f77b4"),
        (axs[1], elevation, mag_xyz[:, 1], "Elevation [deg]", "mag y", mask_el, "#ff7f0e"),
        (axs[2], twist, mag_xyz[:, 2], "Twist [deg]", "mag z", mask_tw, "#2ca02c"),
    ]

    for ax, angle, mag_comp, angle_lbl, mag_lbl, mask, color in rows:
        add_mask_shading(ax, time_s, mask, color)
        l1 = ax.plot(time_s, angle, label=angle_lbl, color=color)
        ax.set_ylabel(angle_lbl)
        ax.grid(linewidth=0.4)
        ax2 = ax.twinx()
        l2 = ax2.plot(time_s, mag_comp, "--", alpha=0.55, color="k", label=mag_lbl)
        ax2.set_ylabel(mag_lbl)
        ax.legend(l1 + l2, [angle_lbl, mag_lbl], loc="upper right")

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{bag_label} - 3-Axis Targets + Segments")
    fig.tight_layout()
    plt.show()


def process_bag(bag_path, arm_index):
    bag_name = os.path.basename(bag_path)
    bag_label = f"{bag_name} (arm{arm_index})"
    print(f"\nBAG_PATH: {bag_path}")
    print(f"ARM_INDEX (mag_mux): {arm_index}")
    print(f"ROTOR_TOPIC setting: {ROTOR_TOPIC}")

    bag = rosbag.Bag(bag_path)
    try:
        mag_topic = resolve_topic_name(bag, MAG_TOPIC)
        joint_topic = resolve_topic_name(bag, JOINT_TOPIC)
        if ROTOR_TOPIC is None:
            rotor_topic = detect_single_rotor_topic(bag)
        else:
            rotor_topic = resolve_topic_name(bag, ROTOR_TOPIC)
        print(f"MAG_TOPIC resolved: {mag_topic}")
        print(f"JOINT_TOPIC resolved: {joint_topic}")
        print(f"ROTOR_TOPIC resolved: {rotor_topic}")

        mag_time, mag_xyz = read_mag_data(bag, mag_topic, arm_index)
        (
            mocap_time,
            joint_raw,
            joint_xyz,
            joint_quat,
            rotor_raw,
            rotor_xyz,
            rotor_quat,
        ) = read_mocap_data(bag, joint_topic, rotor_topic)
    finally:
        bag.close()

    mag_interp = interpolate_xyz(mag_time, mag_xyz, mocap_time)

    az_raw, elevation = calc_azimuth_elevation(joint_xyz, rotor_xyz)
    twist_raw, twist_condition, twist_recalib_used, twist_low_condition_pct = calc_twist_deg(
        mocap_time,
        joint_quat,
        rotor_quat,
        joint_raw,
        rotor_raw,
        return_conditioning=True,
    )
    az_seg = wrap_deg(az_raw)
    twist_seg = wrap_deg(twist_raw)
    twist_unwrapped_preview = unwrap_deg(twist_seg)
    twist_unwrapped_preview -= twist_unwrapped_preview[0]

    az_wraps = int(np.sum(np.abs(np.diff(az_raw)) > 180.0))
    tw_wraps = int(np.sum(np.abs(np.diff(twist_raw)) > 180.0))
    print(f"Samples total: {len(mocap_time)}")
    print(f"Azimuth wrap jumps (raw): {az_wraps}")
    print(f"Twist wrap jumps (raw): {tw_wraps}")
    if twist_recalib_used:
        print(
            "Twist recalibration: enabled "
            f"(low-conditioning {twist_low_condition_pct:.2f}% "
            f">= {TWIST_RECALIB_LOW_CONDITION_PCT:.2f}%)"
        )
    else:
        print(
            "Twist recalibration: not needed "
            f"(low-conditioning {twist_low_condition_pct:.2f}%)"
        )
    print(
        "Twist wrapped range [deg]: "
        f"{float(np.min(twist_seg)):.2f} .. {float(np.max(twist_seg)):.2f}"
    )
    print(
        "Twist unwrapped preview [deg]: "
        f"{float(np.min(twist_unwrapped_preview)):.2f} .. "
        f"{float(np.max(twist_unwrapped_preview)):.2f}"
    )
    low_condition = twist_condition < TWIST_CONDITION_WARN_MIN_NORM
    if np.any(low_condition):
        low_count = int(np.sum(low_condition))
        low_pct = 100.0 * low_count / float(len(low_condition))
        print(
            "Twist conditioning warning: "
            f"{low_count}/{len(low_condition)} samples "
            f"({low_pct:.2f}%) have norm < {TWIST_CONDITION_WARN_MIN_NORM:.3f}"
        )

    azimuth = unwrap_deg(az_seg) if UNWRAP_AZIMUTH else az_seg.copy()
    twist = unwrap_deg(twist_seg) if UNWRAP_TWIST else twist_seg.copy()

    azimuth -= azimuth[0]
    twist -= twist[0]

    mask_az, mask_el, mask_tw, thresholds = detect_axis_masks(
        mocap_time, az_seg, elevation, twist_seg
    )

    print(
        "Segment counts: "
        f"az={int(mask_az.sum())}, el={int(mask_el.sum())}, tw={int(mask_tw.sum())}"
    )
    print(
        "Speed thresholds [deg/s]: "
        f"az={thresholds[0]:.4f}, el={thresholds[1]:.4f}, tw={thresholds[2]:.4f}"
    )

    output_path = build_output_path(bag_path, arm_index)
    with open(output_path, "w", newline="") as lut_file:
        writer = csv.writer(lut_file)
        writer.writerow(["mag_x", "mag_y", "mag_z", "azimuth", "elevation", "twist"])
        for m, az, el, tw in zip(mag_interp, azimuth, elevation, twist):
            writer.writerow([m[0], m[1], m[2], az, el, tw])

    print(f"LUT saved to: {output_path}")

    if SHOW_PLOTS:
        plot_diagnostics(
            mocap_time,
            mag_interp,
            azimuth,
            elevation,
            twist,
            mask_az,
            mask_el,
            mask_tw,
            bag_label,
        )


def main():
    args = parse_args()
    bag_paths = collect_bag_paths(args.bag_input, args.bag_pattern)
    if not bag_paths:
        raise FileNotFoundError(
            f"No bag files matching '{args.bag_pattern}' found in {args.bag_input}"
        )

    print(f"Found {len(bag_paths)} bag file(s).")
    for path in bag_paths:
        if args.arm_index is not None:
            arm_index = args.arm_index
        elif ARM_INDEX is not None:
            arm_index = ARM_INDEX
        else:
            arm_index = infer_arm_index_from_filename(path)
        process_bag(path, arm_index)

    print(f"\nCompleted successfully: {len(bag_paths)} bag(s) processed.")


if __name__ == "__main__":
    main()
