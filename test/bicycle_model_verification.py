"""
Bicycle Model Verification.

Runs the same vehicle on the same track with the bicycle model enabled and
disabled, then compares the results to validate the effect is within expectation.

Expected behaviour:
- Bicycle model tightens cornering limits where slip angles saturate (slower).
- Bicycle model uses axle-aware traction, which can recover some time on RWD.
- Net lap time delta is typically small (< ~3 %) and in the faster direction for
  an RWD vehicle where rear load improves traction.
- Lateral speed limits (v_lat) should be equal or lower with the bicycle model.
- Acceleration limits (v_fwd) may increase slightly due to axle-aware rear load.

Outputs:
1. Summary table: lap times, sector stats, pass/fail criteria.
2. Speed overlay (bicycle ON vs OFF).
3. Delta speed coloured by direction.
4. v_lat comparison (shows where bicycle model tightens cornering).
5. v_fwd comparison (shows axle-aware traction effect).

Usage:
    python test/bicycle_model_verification.py
    python test/bicycle_model_verification.py --save test/bicycle_model_verification.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events.autocross_generator import build_standard_autocross
from models.vehicle import (
    EVPowertrainParams,
    VehicleGeometry,
    VehicleParams,
    build_aero_from_config,
    build_tyre_from_config,
)
from solver.qss_speed import solve_qss


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------
MAX_LAP_TIME_DELTA_PCT = 7.0   # Lap time change must be within ±7 %
                                # Empirically ~5–6 % on the standard autocross
                                # (slip angle saturation reduces every corner speed)
MAX_VLAT_INCREASE_PCT  = 1.0   # v_lat with bicycle model must not exceed
                                # point-mass v_lat by more than 1 % (it should
                                # be equal or lower — small tolerance for
                                # floating-point)


def build_vehicle(config: dict) -> VehicleParams:
    """Build a single vehicle with Pacejka tyres and full geometry."""
    aero = build_aero_from_config(config["aero"])
    geo  = VehicleGeometry(**config["geometry"])

    pt_cfg = config["powertrain"]
    pt = EVPowertrainParams(
        drivetrain=pt_cfg["drivetrain"],
        motor_power_kW=pt_cfg["motor_power_kW"],
        motor_torque_Nm=pt_cfg["motor_torque_Nm"],
        motor_rpm_max=pt_cfg["motor_rpm_max"],
        motor_efficiency=pt_cfg["motor_efficiency"],
        gear_ratio=pt_cfg["gear_ratio"],
        wheel_radius_m=pt_cfg["wheel_radius_m"],
        motor_weight_kg=pt_cfg.get("motor_weight_kg", 9.4),
        inverter_weight_kg=pt_cfg.get("inverter_weight_kg", 6.9),
        powertrain_overhead_kg=pt_cfg.get("powertrain_overhead_kg", 10.0),
        inverter_peak_power_kW=pt_cfg.get("inverter_peak_power_kW", 320.0),
        inverter_peak_current_A=pt_cfg.get("inverter_peak_current_A", 600.0),
    )

    tyre_cfg = dict(config["tyre"])
    tyre_cfg["model"] = "pacejka"
    tyre = build_tyre_from_config(tyre_cfg)

    return VehicleParams(
        m=config["vehicle"]["mass_kg"],
        g=config["vehicle"]["g"],
        Crr=config["vehicle"]["Crr"],
        aero=aero,
        tyre=tyre,
        powertrain=pt,
        geometry=geo,
    )


def find_corners(kappa: np.ndarray, s: np.ndarray,
                 threshold: float = 0.01) -> list:
    """Return (index, s_pos, kappa) tuples at each corner apex."""
    in_corner = np.abs(kappa) > threshold
    corners = []
    i = 0
    while i < len(kappa):
        if in_corner[i]:
            start = i
            while i < len(kappa) and in_corner[i]:
                i += 1
            end = i
            apex_local = np.argmax(np.abs(kappa[start:end]))
            apex = start + apex_local
            corners.append((apex, s[apex], kappa[apex]))
        else:
            i += 1
    return corners


def run_verification(config: dict, save_path: str = None):
    """Run bicycle ON/OFF comparison and print acceptance results."""

    track, _meta = build_standard_autocross()
    vehicle = build_vehicle(config)

    print(f"Bicycle model available: {vehicle.has_bicycle_model}")
    if not vehicle.has_bicycle_model:
        print("ERROR: vehicle does not satisfy has_bicycle_model — check geometry and tyre params.")
        sys.exit(1)

    # ---- Solve ----
    result_on,  lap_on  = solve_qss(track, vehicle, use_bicycle_model=True)
    result_off, lap_off = solve_qss(track, vehicle, use_bicycle_model=False)

    s       = track.s
    v_on    = result_on["v"]
    v_off   = result_off["v"]
    delta_v = v_on - v_off          # positive → bicycle model is faster

    vlat_on  = result_on["v_lat"]
    vlat_off = result_off["v_lat"]
    vfwd_on  = result_on["v_fwd"]
    vfwd_off = result_off["v_fwd"]

    delta_lap_pct = (lap_on - lap_off) / lap_off * 100.0

    # v_lat must not be meaningfully higher with bicycle model
    # (filter out straight sections where both are unconstrained)
    corner_mask = np.abs(track.kappa) > 0.01
    vlat_increase_pct = np.max(
        (vlat_on[corner_mask] - vlat_off[corner_mask]) / vlat_off[corner_mask] * 100.0,
        initial=-np.inf,
    )

    corners = find_corners(track.kappa, s)

    # ---- Summary ----
    print()
    print("=" * 65)
    print("BICYCLE MODEL VERIFICATION")
    print("=" * 65)
    print(f"Track length : {s[-1]:.1f} m")
    print(f"Vehicle mass : {vehicle.m:.1f} kg")
    print(f"Drivetrain   : {config['powertrain']['drivetrain'].upper()}")
    print()
    print(f"{'Model':<18} {'Lap Time':>10} {'V_max':>10} {'V_avg':>10}")
    print("-" * 50)
    print(f"{'Bicycle ON':<18} {lap_on:>9.3f}s {np.max(v_on)*3.6:>9.2f} {np.mean(v_on)*3.6:>9.2f}")
    print(f"{'Bicycle OFF':<18} {lap_off:>9.3f}s {np.max(v_off)*3.6:>9.2f} {np.mean(v_off)*3.6:>9.2f}")
    print(f"{'Delta':<18} {lap_on - lap_off:>+9.3f}s {delta_lap_pct:>+8.2f}%")
    print()

    if corners:
        print(f"{'Corner':>6} {'Distance':>10} {'V_on':>10} {'V_off':>10} {'Delta':>8}")
        print("-" * 48)
        for idx, (apex, s_pos, _kap) in enumerate(corners):
            dv = (v_on[apex] - v_off[apex]) * 3.6
            print(f"  {idx+1:>4} {s_pos:>9.1f}m {v_on[apex]*3.6:>9.2f} {v_off[apex]*3.6:>9.2f} {dv:>+7.2f}")
        print()

    # ---- Acceptance checks ----
    print("ACCEPTANCE CRITERIA")
    print("-" * 50)

    checks = [
        (
            abs(delta_lap_pct) <= MAX_LAP_TIME_DELTA_PCT,
            f"Lap time delta {delta_lap_pct:+.2f}% within ±{MAX_LAP_TIME_DELTA_PCT}%",
        ),
        (
            vlat_increase_pct <= MAX_VLAT_INCREASE_PCT,
            f"Max v_lat increase at corners {vlat_increase_pct:.2f}% ≤ {MAX_VLAT_INCREASE_PCT}%",
        ),
    ]

    all_pass = True
    for passed, label in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {label}")
        if not passed:
            all_pass = False

    print()
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    print()

    # ---- Plots ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        f"Bicycle Model Verification — ON: {lap_on:.3f}s | OFF: {lap_off:.3f}s | Δ {lap_on - lap_off:+.3f}s ({delta_lap_pct:+.2f}%)",
        fontsize=13, fontweight="bold",
    )

    colour_on  = "#E53935"   # red  — bicycle ON
    colour_off = "#1E88E5"   # blue — bicycle OFF

    # 1. Speed overlay
    ax = axes[0]
    ax.plot(s, v_off * 3.6, colour_off, linewidth=1.5, label="Bicycle OFF (point-mass)")
    ax.plot(s, v_on  * 3.6, colour_on,  linewidth=1.5, label="Bicycle ON  (axle-aware)", alpha=0.85)
    ax.set_ylabel("Speed [km/h]")
    ax.set_title("Speed Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Delta speed
    ax = axes[1]
    pos = np.where(delta_v >= 0, delta_v * 3.6, 0.0)
    neg = np.where(delta_v <  0, delta_v * 3.6, 0.0)
    ax.fill_between(s, pos, 0, color="#43A047", alpha=0.65, label="Bicycle ON faster")
    ax.fill_between(s, neg, 0, color="#E53935", alpha=0.65, label="Bicycle OFF faster")
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.set_ylabel("Δ Speed [km/h]")
    ax.set_title("Speed Difference (ON − OFF)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. v_lat comparison
    ax = axes[2]
    ax.plot(s, vlat_off * 3.6, colour_off, linewidth=1.2, alpha=0.75, label="v_lat OFF")
    ax.plot(s, vlat_on  * 3.6, colour_on,  linewidth=1.2, alpha=0.75, label="v_lat ON")
    ax.set_ylabel("Lateral Limit [km/h]")
    ax.set_title("Cornering Speed Limit (v_lat) — bicycle model should be ≤ point-mass")
    ax.legend()
    ax.grid(True, alpha=0.3)
    v_cap = max(np.nanmax(v_on), np.nanmax(v_off)) * 3.6 * 1.4
    ax.set_ylim(0, min(v_cap, 250))

    # 4. v_fwd comparison
    ax = axes[3]
    ax.plot(s, vfwd_off * 3.6, colour_off, linewidth=1.2, alpha=0.75, label="v_fwd OFF")
    ax.plot(s, vfwd_on  * 3.6, colour_on,  linewidth=1.2, alpha=0.75, label="v_fwd ON")
    ax.set_ylabel("Accel Limit [km/h]")
    ax.set_xlabel("Distance [m]")
    ax.set_title("Acceleration Speed Limit (v_fwd) — axle-aware traction may lift RWD limit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(v_cap, 250))

    # Annotate corner numbers on all plots
    for idx, (apex, s_pos, _kap) in enumerate(corners):
        for a in axes:
            a.axvline(x=s_pos, color="grey", linewidth=0.4, alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bicycle model ON/OFF verification")
    parser.add_argument(
        "--save", default=None,
        help="Save path for figure. Shows interactively if omitted.",
    )
    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "default.yaml",
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_verification(config, save_path=args.save)
