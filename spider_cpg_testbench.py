#!/usr/bin/env python3
"""Generate a spider-like MuJoCo MJCF test bench for CPG experiments.

This script creates a self-contained MJCF/XML model with:
- 6 legs in a hexagonal arrangement (60 degrees between adjacent leg bases)
- 3 DOF per leg
- 5 passive/fixed legs
- one active leg whose foot follows an elliptical trajectory in a plane parallel
  to the body side, using a simple CPG-inspired sinusoid

The user asked for a .mcjf file, which is a typo for MuJoCo MJCF. The generated
output file therefore uses the standard MJCF XML format and is saved with a .mcjf
extension to match the request.
"""

from __future__ import annotations

import argparse
from math import cos, pi, sin
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# Simpler, cleaner body/leg proportions inspired by a compact spider robot.
BODY_RADIUS = 0.17
BODY_HALF_EXTENTS = (0.18, 0.12, 0.04)
LEG_COXA_LEN = 0.05
LEG_FEMUR_LEN = 0.10
LEG_TIBIA_LEN = 0.12

# Symmetric six-leg arrangement with three equally spaced hips on each side:
# left side: -150°, -90°, -30°
# right side: 30°, 90°, 150°
# This keeps the body mirror-symmetric and distributes 3 legs on each side.
LEG_THETAS_RAD = [
    -5.0 * pi / 6.0,
    -pi / 2.0,
    -pi / 6.0,
    pi / 6.0,
    pi / 2.0,
    5.0 * pi / 6.0,
]


def make_leg_blocks(active_leg: int | None = 0) -> str:
    """Return XML for all six legs. One leg is active, the rest are fixed.

    Legend:
    - hip_yaw: yaw about the body-local Z axis
    - hip_pitch: pitch forward/backward in the radial plane
    - knee_pitch: flexion/extension of the distal segment
    """
    blocks: list[str] = []

    for i in range(6):
        theta = LEG_THETAS_RAD[i]
        x = BODY_RADIUS * cos(theta)
        y = BODY_RADIUS * sin(theta)
        leg_name = f"leg_{i}"
        _is_active = i == active_leg

        blocks.append(
            f"""
    <body name="{leg_name}_base" pos="{x:.6f} {y:.6f} 0.0" euler="0 0 {theta:.6f}">
      <joint name="{leg_name}_yaw" type="hinge" axis="0 0 1" range="-1.5708 1.5708" damping="0.03" stiffness="0.0"/>
      <geom type="capsule" fromto="0 0 0 {LEG_COXA_LEN:.6f} 0 0" size="0.010" rgba="0.72 0.72 0.72 1"/>
      <body name="{leg_name}_femur" pos="{LEG_COXA_LEN:.6f} 0 0">
        <joint name="{leg_name}_pitch" type="hinge" axis="0 1 0" range="-1.5708 1.5708" damping="0.03" stiffness="0.0"/>
        <geom type="capsule" fromto="0 0 0 0 {LEG_FEMUR_LEN:.6f} 0" size="0.008" rgba="0.68 0.68 0.68 1"/>
        <body name="{leg_name}_tibia" pos="0 {LEG_FEMUR_LEN:.6f} 0">
          <joint name="{leg_name}_knee" type="hinge" axis="0 1 0" range="-2.0944 0.0" damping="0.03" stiffness="0.0"/>
          <geom type="capsule" fromto="0 0 0 0 {LEG_TIBIA_LEN:.6f} 0" size="0.007" rgba="0.62 0.62 0.62 1"/>
          <site name="{leg_name}_foot" pos="0 {LEG_TIBIA_LEN:.6f} 0" size="0.004"/>
        </body>
      </body>
    </body>
            """.strip()
        )

    return "\n".join(blocks)


def make_actuators(active_leg: int) -> str:
    """Create position actuators for the active leg and zero-motion controls for others."""
    actuators: list[str] = []
    for i in range(6):
        leg_name = f"leg_{i}"
        for joint in ("yaw", "pitch", "knee"):
            if i == active_leg:
                kp = 70.0
                ctrl_range = "-1.57 1.57" if joint == "yaw" else "-2.09 2.09"
            else:
                kp = 300.0
                ctrl_range = "-0.01 0.01"
            actuators.append(
                f'<position name="{leg_name}_{joint}_act" joint="{leg_name}_{joint}" kp="{kp:.1f}" ctrlrange="{ctrl_range}"/>'
            )
    return "\n    ".join(actuators)


def cpg_elliptic_targets(time_s: float, phase_offset: float = 0.0) -> tuple[float, float, float]:
    """Return target angles for the active leg's 3 joints.

    The leg tip follows an ellipse in a plane parallel to the body side.
    """
    yaw = 0.12 * sin(2 * pi * 0.6 * time_s + phase_offset)
    pitch = 0.60 * sin(2 * pi * 0.8 * time_s + phase_offset)
    knee = 0.45 * sin(2 * pi * 0.8 * time_s + phase_offset + pi / 2.0)
    return yaw, pitch, knee


def generate_mjcf(active_leg: int = 0) -> str:
    leg_xml = make_leg_blocks(active_leg=active_leg)
    actuators_xml = make_actuators(active_leg=active_leg)
    return f'''<mujoco model="spider_cpg_testbench">
  <compiler angle="radian" inertiafromgeom="auto"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>

  <asset>
    <material name="body_mat" rgba="0.25 0.55 0.95 1"/>
  </asset>

  <default>
    <joint damping="0.05" armature="0.01" stiffness="0.0"/>
    <geom condim="3" friction="1.0 0.05 0.01" density="1000"/>
  </default>

  <worldbody>
    <body name="root" pos="0 0 0.35">
      <inertial pos="0 0 0" mass="1.6" diaginertia="0.2 0.2 0.2"/>
      <geom type="box" size="{BODY_HALF_EXTENTS[0]} {BODY_HALF_EXTENTS[1]} {BODY_HALF_EXTENTS[2]}" material="body_mat"/>
      {leg_xml}
    </body>
  </worldbody>

  <actuator>
    {actuators_xml}
  </actuator>
</mujoco>
'''


def write_model(output_path: str | Path, active_leg: int = 0) -> Path:
    path = Path(output_path)
    path.write_text(generate_mjcf(active_leg=active_leg), encoding="utf-8")
    return path


def set_cpg_controls(data: mujoco.MjData, model: mujoco.MjModel, active_leg: int, time_s: float) -> None:
    """Apply periodic CPG targets to one leg and zero the others."""
    yaw, pitch, knee = cpg_elliptic_targets(time_s)
    ctrl = np.zeros(model.nu)
    base = 3 * active_leg
    ctrl[base: base + 3] = np.array([yaw, pitch, knee], dtype=float)
    data.ctrl[:] = ctrl


def run_viewer(active_leg: int = 0, output_path: str | Path | None = None, seconds: float = 10.0) -> None:
    """Generate the model, open MuJoCo viewer, and animate the active leg."""
    if output_path is None:
        output_path = Path(__file__).with_name("spider_cpg_testbench.mcjf")
    write_model(output_path, active_leg=active_leg)
    model = mujoco.MjModel.from_xml_path(str(output_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Opening MuJoCo viewer for spider CPG bench. Active leg: leg_{active_leg}")
    print("Press ESC in the viewer to close.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = data.time
        while viewer.is_running() and data.time - start < seconds:
            set_cpg_controls(data, model, active_leg, data.time)
            mujoco.mj_step(model, data)
            viewer.sync()


def main() -> None:
    ap = argparse.ArgumentParser(description="Spider CPG test-bench generator and viewer")
    ap.add_argument("--show", action="store_true", help="open the MuJoCo viewer and animate the CPG leg")
    ap.add_argument("--active-leg", type=int, default=0, help="Which leg is the CPG-driven active leg (0-5)")
    ap.add_argument("--seconds", type=float, default=10.0, help="Runtime for the viewer animation in seconds")
    ap.add_argument("--output", type=str, default="spider_cpg_testbench.mcjf", help="MJCF output path")
    args = ap.parse_args()

    output_file = Path(args.output)
    write_model(output_file, active_leg=args.active_leg)
    print(f"Wrote spider model to: {output_file}")
    print("The active leg is the one with the CPG-driven sinusoidal motion; the others remain at zero pose.")

    if args.show:
        run_viewer(active_leg=args.active_leg, output_path=output_file, seconds=args.seconds)
    else:
        print("Run with --show to open the MuJoCo viewer and animate the CPG leg.")


if __name__ == "__main__":
    main()
