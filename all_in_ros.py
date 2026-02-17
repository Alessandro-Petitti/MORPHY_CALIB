#!/usr/bin/env python3
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import rosbag

# =========================
# USER SETTINGS (edit these)
# =========================
BAG_PATH = os.path.join("bag/arm_3", "calib_arm_3.bag")
ARM_INDEX = 3
ROTOR_TOPIC = None  # Auto-detect if None (expects one /mocap/arm*/pose in bag)

# Fixed topics
MAG_TOPIC = "/mavros/mag_mux/raw"
JOINT_TOPIC = "/mocap/body/pose"

# Behavior / output
SHOW_PLOTS = True
INVERT_YZ = True
UNWRAP_AZIMUTH = True
ROTOR_Z_OFFSET = 6.499424548581256 / 1000
OUTPUT_DIR = os.path.join("data", "lut")


def calc_angles(joint_xyz, rotor_xyz):
    delta = rotor_xyz - joint_xyz
    x, y, z = delta[:, 0], delta[:, 1], delta[:, 2]
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x * x + y * y))
    return np.rad2deg(azimuth), np.rad2deg(elevation)


def unwrap_deg(angle_deg):
    return np.rad2deg(np.unwrap(np.deg2rad(angle_deg)))


def transform_point(x, y, z, z_offset=0.0):
    z_value = z - z_offset
    if INVERT_YZ:
        return [x, -y, -z_value]
    return [x, y, z_value]


def interpolate_xyz(source_time, source_xyz, target_time):
    if len(source_time) < 2:
        raise RuntimeError("Not enough samples to interpolate.")
    out = np.zeros((len(target_time), 3))
    for i in range(3):
        out[:, i] = np.interp(target_time, source_time, source_xyz[:, i])
    return out


def resolve_topic_name(bag, requested_topic):
    topics = set(bag.get_type_and_topic_info().topics.keys())
    if requested_topic in topics:
        return requested_topic

    alt = requested_topic[1:] if requested_topic.startswith("/") else "/" + requested_topic
    if alt in topics:
        return alt

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


def read_mag_data(bag, mag_topic):
    mag_time = []
    mag_xyz = []

    for _, msg, _ in bag.read_messages(topics=[mag_topic]):
        if len(msg.mags) <= ARM_INDEX:
            continue
        mag = msg.mags[ARM_INDEX]
        mag_time.append(msg.header.stamp.to_sec())
        mag_xyz.append([mag.x, mag.y, mag.z])

    if not mag_time:
        raise RuntimeError(
            f"No valid samples on {mag_topic} for mag_mux[{ARM_INDEX}]."
        )
    return np.asarray(mag_time), np.asarray(mag_xyz)


def read_pose_topic(bag, topic, z_offset):
    pose_time = []
    pose_xyz = []

    for _, msg, _ in bag.read_messages(topics=[topic]):
        p = msg.pose.position
        pose_time.append(msg.header.stamp.to_sec())
        pose_xyz.append(transform_point(p.x, p.y, p.z, z_offset=z_offset))

    if not pose_time:
        raise RuntimeError(f"No pose samples found on {topic}.")
    return np.asarray(pose_time), np.asarray(pose_xyz)


def read_mocap_data(bag, joint_topic, rotor_topic):
    joint_time, joint_xyz_raw = read_pose_topic(bag, joint_topic, z_offset=0.0)
    rotor_time, rotor_xyz = read_pose_topic(
        bag,
        rotor_topic,
        z_offset=ROTOR_Z_OFFSET,
    )
    joint_xyz = interpolate_xyz(joint_time, joint_xyz_raw, rotor_time)
    return rotor_time, joint_xyz, rotor_xyz


def build_output_path():
    bag_stem = os.path.splitext(os.path.basename(BAG_PATH))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"{bag_stem}_arm{ARM_INDEX}.csv")


def plot_data(joint_xyz, rotor_xyz, mocap_time, mag_time, mag_xyz, azimuth, elevation):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(joint_xyz[:, 0], label="joint x")
    plt.plot(rotor_xyz[:, 0], label="rotor x")
    plt.grid(linewidth=0.4)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(joint_xyz[:, 1], label="joint y")
    plt.plot(rotor_xyz[:, 1], label="rotor y")
    plt.grid(linewidth=0.4)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(joint_xyz[:, 2], label="joint z")
    plt.plot(rotor_xyz[:, 2], label="rotor z")
    plt.grid(linewidth=0.4)
    plt.legend()
    plt.tight_layout()

    _, ax = plt.subplots(nrows=2)
    ax[0].plot(mocap_time, azimuth, label="azimuth")
    ax[0].plot(mag_time, mag_xyz[:, 0], "--", label="mag x")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(mocap_time, elevation, label="elevation")
    ax[1].plot(mag_time, mag_xyz[:, 1], "--", label="mag y")
    ax[1].grid()
    ax[1].legend()
    plt.tight_layout()
    plt.show()


def main():
    print(f"BAG_PATH: {BAG_PATH}")
    print(f"ARM_INDEX (mag_mux): {ARM_INDEX}")
    print(f"ROTOR_TOPIC setting: {ROTOR_TOPIC}")

    bag = rosbag.Bag(BAG_PATH)
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

        mag_time, mag_xyz = read_mag_data(bag, mag_topic)
        mocap_time, joint_xyz, rotor_xyz = read_mocap_data(
            bag,
            joint_topic,
            rotor_topic,
        )
    finally:
        bag.close()

    azimuth, elevation = calc_angles(joint_xyz, rotor_xyz)
    wrap_jumps = int(np.sum(np.abs(np.diff(azimuth)) > 180.0))
    print(f"Azimuth wrap jumps (raw): {wrap_jumps}")
    if UNWRAP_AZIMUTH:
        azimuth = unwrap_deg(azimuth)
    azimuth -= azimuth[0]
    mag_resampled = interpolate_xyz(mag_time, mag_xyz, mocap_time)

    output_path = build_output_path()
    with open(output_path, "w", newline="") as lut_file:
        writer = csv.writer(lut_file)
        writer.writerow(["mag_x", "mag_y", "mag_z", "azimuth", "elevation"])
        for mag_xyz_row, az, el in zip(mag_resampled, azimuth, elevation):
            writer.writerow([mag_xyz_row[0], mag_xyz_row[1], mag_xyz_row[2], az, el])

    print(f"LUT saved to: {output_path}")

    if SHOW_PLOTS:
        plot_data(
            joint_xyz,
            rotor_xyz,
            mocap_time,
            mag_time,
            mag_xyz,
            azimuth,
            elevation,
        )


if __name__ == "__main__":
    main()
