#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================
# USER SETTINGS
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
LUT_DIR = SCRIPT_DIR / "data" / "lut"
SHOW_POINTS = False

REQUIRED_COLS = ["mag_x", "mag_y", "mag_z", "azimuth", "elevation"]


def unwrap_deg(angle_deg):
    return np.rad2deg(np.unwrap(np.deg2rad(angle_deg)))


def r2_score(y_true, y_pred):
    u = np.sum((y_true - y_pred) ** 2)
    v = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - u / v if v != 0 else 0.0


def fit_line(x, y):
    X = np.c_[np.ones(x.shape[0]), x]
    return np.linalg.pinv(X) @ y


def apply_line(x, beta):
    X = np.c_[np.ones(x.shape[0]), x]
    return X @ beta


def fit_circular_angle(x, angle_deg):
    # Circular regression: fit sin(theta) and cos(theta), then rebuild theta.
    theta = np.deg2rad(angle_deg)
    y_sin = np.sin(theta)
    y_cos = np.cos(theta)
    b_sin = fit_line(x, y_sin)
    b_cos = fit_line(x, y_cos)
    sin_pred = apply_line(x, b_sin)
    cos_pred = apply_line(x, b_cos)
    pred_wrapped = np.rad2deg(np.arctan2(sin_pred, cos_pred))
    pred = unwrap_deg(pred_wrapped)
    pred += np.median(angle_deg - pred)
    return pred, r2_score(angle_deg, pred), b_sin, b_cos


def main():
    csv_files = sorted(LUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {LUT_DIR.resolve()}")

    print(f"Found {len(csv_files)} LUT files in {LUT_DIR.resolve()}")
    for p in csv_files:
        print(" -", p.name)

    n = len(csv_files)
    fig, axs = plt.subplots(nrows=n, ncols=1, figsize=(12, max(4, 4 * n)))
    if n == 1:
        axs = [axs]

    for ax, file in zip(axs, csv_files):
        df = pd.read_csv(file)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        unexpected = [c for c in df.columns if c not in REQUIRED_COLS]
        if missing or unexpected:
            raise ValueError(
                f"{file.name} invalid columns. Missing: {missing}; Unexpected: {unexpected}"
            )

        A_az = df[["mag_x", "mag_z"]].to_numpy()
        A_el = df[["mag_y", "mag_z"]].to_numpy()
        azimuth = unwrap_deg(df["azimuth"].to_numpy())
        elevation = df["elevation"].to_numpy()
        mag_x = df["mag_x"].to_numpy()
        mag_y = df["mag_y"].to_numpy()

        az_pred, r2_az, b_sin_az, b_cos_az = fit_circular_angle(A_az, azimuth)
        b_el = fit_line(A_el, elevation)
        el_pred = apply_line(A_el, b_el)
        r2_el = r2_score(elevation, el_pred)

        wrap_jumps = int(np.sum(np.abs(np.diff(df["azimuth"].to_numpy())) > 180.0))

        print(f"\n{file.name}")
        print(f"  Azimuth wrap jumps raw: {wrap_jumps}")
        print(f"  Azimuth R^2   : {r2_az:.4f}")
        print(f"  Elevation R^2 : {r2_el:.4f}")
        print(f"  Az sin params  : {tuple(b_sin_az)}")
        print(f"  Az cos params  : {tuple(b_cos_az)}")
        print(f"  El params [rad]: {tuple(np.radians(b_el))}")

        if SHOW_POINTS:
            ax.scatter(mag_x, azimuth, s=4, alpha=0.5, label="Azimuth")
            ax.scatter(mag_y, elevation, s=4, alpha=0.5, label="Elevation")
        else:
            ax.plot(mag_x, azimuth, label="Azimuth")
            ax.plot(mag_y, elevation, label="Elevation")

        ax.plot(mag_x, az_pred, "--", c="k", label=f"Azimuth fit (R^2={r2_az:.2f})")
        ax.plot(
            mag_y,
            el_pred,
            ":",
            c="k",
            label=f"Elevation fit (R^2={r2_el:.2f})",
        )

        ax.set_title(file.name)
        ax.set_xlabel("Magnetic Field [mT]")
        ax.set_ylabel("Deflection Angle [deg]")
        ax.grid()
        ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
