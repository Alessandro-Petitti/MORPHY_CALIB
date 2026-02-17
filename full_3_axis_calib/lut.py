#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================
# USER SETTINGS
# =========================
LUT_DIR = Path("data/lut_3axis")
SHOW_POINTS = False
MIN_SEGMENT_LEN_AZ = 40
MIN_SEGMENT_LEN_EL = 40
MIN_SEGMENT_LEN_TW = 8
SPEED_PCTILE_AZ = 75.0
SPEED_PCTILE_EL = 75.0
SPEED_PCTILE_TW = 60.0
DOMINANCE_RATIO_MIN_AZ = 1.35
DOMINANCE_RATIO_MIN_EL = 1.35
DOMINANCE_RATIO_MIN_TW = 1.10
SPEED_SMOOTH_WINDOW = 21
TWIST_TARGET_MODE = "wrapped"  # "wrapped" or "unwrapped"
PROTOCOL_PHASE_MODE = "auto"  # "off", "auto", "force"
PROTOCOL_ACTIVE_PCTILE = 55.0
PROTOCOL_AXIS_PCTILE = 60.0
PROTOCOL_MIN_ACTIVE_LEN = 120

REQUIRED_COLS = ["mag_x", "mag_y", "mag_z", "azimuth", "elevation", "twist"]
EPS = 1e-9


def unwrap_deg(angle_deg):
    return np.rad2deg(np.unwrap(np.deg2rad(angle_deg)))


def wrap_deg(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def r2_score(y_true, y_pred):
    u = np.sum((y_true - y_pred) ** 2)
    v = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - u / v if v != 0 else 0.0


def circular_r2_score(y_true_deg, y_pred_deg):
    t_true = np.deg2rad(y_true_deg)
    t_pred = np.deg2rad(y_pred_deg)
    r2_sin = r2_score(np.sin(t_true), np.sin(t_pred))
    r2_cos = r2_score(np.cos(t_true), np.cos(t_pred))
    return 0.5 * (r2_sin + r2_cos)


def smooth_series(x, window):
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")


def fit_linear(X, y):
    A = np.c_[np.ones(X.shape[0]), X]
    return np.linalg.pinv(A) @ y


def predict_linear(X, beta):
    A = np.c_[np.ones(X.shape[0]), X]
    return A @ beta


def fit_circular(X_fit, angle_deg_fit, X_pred):
    theta = np.deg2rad(angle_deg_fit)
    y_sin = np.sin(theta)
    y_cos = np.cos(theta)
    b_sin = fit_linear(X_fit, y_sin)
    b_cos = fit_linear(X_fit, y_cos)

    sin_pred = predict_linear(X_pred, b_sin)
    cos_pred = predict_linear(X_pred, b_cos)
    pred_wrapped = np.rad2deg(np.arctan2(sin_pred, cos_pred))
    return pred_wrapped, b_sin, b_cos


def poly2_features(X):
    n = X.shape[1]
    parts = [X]
    for i in range(n):
        parts.append((X[:, i] ** 2).reshape(-1, 1))
    for i in range(n):
        for j in range(i + 1, n):
            parts.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.column_stack(parts)


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


def detect_axis_masks(azimuth, elevation, twist):
    def circular_speed_deg_sample(angle_deg):
        if len(angle_deg) < 2:
            return np.zeros_like(angle_deg, dtype=float)
        dtheta = wrap_deg(np.diff(angle_deg))
        speed = np.empty(len(angle_deg), dtype=float)
        speed[1:] = np.abs(dtheta)
        speed[0] = speed[1]
        return speed

    daz = circular_speed_deg_sample(azimuth)
    delv = np.abs(np.gradient(elevation))
    dtw = circular_speed_deg_sample(twist)

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


def contiguous_spans(mask):
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
        spans.append((start, end))
    return spans


def add_mask_shading(ax, mask, color):
    for i0, i1 in contiguous_spans(mask):
        ax.axvspan(i0, i1, color=color, alpha=0.10, linewidth=0.0)


def index_or_full(mask, min_len):
    idx = np.flatnonzero(mask)
    if len(idx) >= min_len:
        return idx, False
    return np.arange(len(mask)), True


def fit_twist_best_model(X_fit, twist_fit, X_full, twist_full, target_mode):
    tw_pred_circ_wrapped, bsin_tw, bcos_tw = fit_circular(X_fit, twist_fit, X_full)

    b_lin = fit_linear(X_fit, twist_fit)
    tw_pred_lin = predict_linear(X_full, b_lin)

    X2_fit = poly2_features(X_fit)
    X2_full = poly2_features(X_full)
    b_poly2 = fit_linear(X2_fit, twist_fit)
    tw_pred_poly2 = predict_linear(X2_full, b_poly2)

    if target_mode == "wrapped":
        twist_target = wrap_deg(twist_full)
        tw_pred_circ = wrap_deg(tw_pred_circ_wrapped)
        tw_pred_lin = wrap_deg(tw_pred_lin)
        tw_pred_poly2 = wrap_deg(tw_pred_poly2)
        # Use regular R² for small angular ranges: circular_r2_score is
        # misleading when cos(θ)≈1 (near-zero variance → R²_cos≈0).
        angle_range = float(np.ptp(twist_target))
        if angle_range < 90.0:
            score_fn = r2_score
            metric_name = "r2 (small-angle)"
        else:
            score_fn = circular_r2_score
            metric_name = "circular_r2"
    elif target_mode == "unwrapped":
        twist_target = twist_full
        tw_pred_circ = unwrap_deg(tw_pred_circ_wrapped)
        tw_pred_circ += np.median(twist_target - tw_pred_circ)
        score_fn = r2_score
        metric_name = "r2"
    else:
        raise ValueError(f"Unsupported TWIST_TARGET_MODE: {target_mode}")

    r2_circ = score_fn(twist_target, tw_pred_circ)
    r2_lin = score_fn(twist_target, tw_pred_lin)
    r2_poly2 = score_fn(twist_target, tw_pred_poly2)

    candidates = {
        "circular": (r2_circ, tw_pred_circ),
        "linear": (r2_lin, tw_pred_lin),
        "poly2": (r2_poly2, tw_pred_poly2),
    }
    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_r2, best_pred = candidates[best_name]
    params = {
        "circ_sin": bsin_tw,
        "circ_cos": bcos_tw,
        "linear": b_lin,
        "poly2": b_poly2,
        "r2_circ": r2_circ,
        "r2_lin": r2_lin,
        "r2_poly2": r2_poly2,
        "best": best_name,
        "best_r2": best_r2,
        "metric": metric_name,
    }
    return best_pred, params


def main():
    csv_files = sorted(LUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {LUT_DIR.resolve()}")

    print(f"Found {len(csv_files)} LUT files in {LUT_DIR.resolve()}")
    for p in csv_files:
        print(" -", p.name)

    for file in csv_files:
        df = pd.read_csv(file)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{file.name} missing columns: {missing}")

        X = df[["mag_x", "mag_y", "mag_z"]].to_numpy()
        az_raw = df["azimuth"].to_numpy()
        elevation = df["elevation"].to_numpy()
        tw_raw = df["twist"].to_numpy()
        azimuth = unwrap_deg(az_raw)
        azimuth_seg = wrap_deg(az_raw)
        twist_wrapped = wrap_deg(tw_raw)
        if TWIST_TARGET_MODE == "wrapped":
            twist = twist_wrapped
        elif TWIST_TARGET_MODE == "unwrapped":
            twist = unwrap_deg(tw_raw)
        else:
            raise ValueError(f"Unsupported TWIST_TARGET_MODE: {TWIST_TARGET_MODE}")

        mask_az, mask_el, mask_tw, thresholds = detect_axis_masks(
            azimuth_seg, elevation, twist_wrapped
        )

        # Keep azimuth/elevation on full-data for stability.
        idx_az = np.arange(len(df))
        idx_el = np.arange(len(df))
        fallback_az = False
        fallback_el = False
        idx_tw, fallback_tw = index_or_full(mask_tw, MIN_SEGMENT_LEN_TW)

        az_pred_wrapped, bsin_az, bcos_az = fit_circular(X[idx_az], azimuth[idx_az], X)
        az_pred = unwrap_deg(az_pred_wrapped)
        az_pred += np.median(azimuth - az_pred)
        r2_az = r2_score(azimuth, az_pred)

        bel = fit_linear(X[idx_el], elevation[idx_el])
        el_pred = predict_linear(X, bel)
        r2_el = r2_score(elevation, el_pred)

        # Twist fit: uses only mag as input (no cascaded az/el predictions)
        # so the model is deployable as a simple f(mag_x, mag_y, mag_z) → twist.
        tw_pred, tw_model = fit_twist_best_model(
            X[idx_tw], twist[idx_tw], X, twist, TWIST_TARGET_MODE
        )
        r2_tw = tw_model["best_r2"]

        az_wraps = int(np.sum(np.abs(np.diff(az_raw)) > 180.0))
        tw_wraps = int(np.sum(np.abs(np.diff(tw_raw)) > 180.0))

        print(f"\n{file.name}")
        print(
            "  Segment samples: "
            f"az={int(mask_az.sum())}, el={int(mask_el.sum())}, tw={int(mask_tw.sum())}"
        )
        print(
            "  Segment thresholds [deg/sample]: "
            f"az={thresholds[0]:.4f}, el={thresholds[1]:.4f}, tw={thresholds[2]:.4f}"
        )
        print(f"  Twist target mode: {TWIST_TARGET_MODE}")
        print(f"  Twist score metric: {tw_model['metric']}")
        print(f"  Fallback full-data: az={fallback_az}, el={fallback_el}, tw={fallback_tw}")
        print(f"  Wrap jumps raw: az={az_wraps}, tw={tw_wraps}")
        print(f"  R^2 azimuth   : {r2_az:.4f}")
        print(f"  R^2 elevation : {r2_el:.4f}")
        print(f"  R^2 twist     : {r2_tw:.4f}")
        print(f"  Az sin params : {tuple(bsin_az)}")
        print(f"  Az cos params : {tuple(bcos_az)}")
        print(f"  El params     : {tuple(bel)}")
        print(
            "  Twist model candidates R^2: "
            f"circular={tw_model['r2_circ']:.4f}, "
            f"linear={tw_model['r2_lin']:.4f}, "
            f"poly2={tw_model['r2_poly2']:.4f}"
        )
        print(f"  Twist model selected: {tw_model['best']}")
        print(f"  Tw sin params : {tuple(tw_model['circ_sin'])}")
        print(f"  Tw cos params : {tuple(tw_model['circ_cos'])}")
        print(f"  Tw linear params : {tuple(tw_model['linear'])}")
        print(f"  Tw poly2 params : {tuple(tw_model['poly2'])}")

        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(14, 9))
        x_idx = np.arange(len(df))

        rows = [
            (axs[0], azimuth, az_pred, "Azimuth [deg]", mask_az, "#1f77b4", r2_az),
            (axs[1], elevation, el_pred, "Elevation [deg]", mask_el, "#ff7f0e", r2_el),
            (axs[2], twist, tw_pred, "Twist [deg]", mask_tw, "#2ca02c", r2_tw),
        ]

        for ax, y, yhat, label, mask, color, r2 in rows:
            add_mask_shading(ax, mask, color)
            if SHOW_POINTS:
                ax.scatter(x_idx, y, s=5, alpha=0.5, label=f"{label} data")
            else:
                ax.plot(x_idx, y, label=f"{label} data")
            ax.plot(x_idx, yhat, "--", c="k", label=f"{label} fit (R^2={r2:.2f})")
            ax.set_ylabel(label)
            ax.grid()
            ax.legend()

        axs[-1].set_xlabel("Sample index")
        fig.suptitle(file.name)
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
