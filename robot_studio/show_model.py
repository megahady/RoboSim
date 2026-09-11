"""Show a robot model in the native MuJoCo viewer with a studio look.

Usage (macOS needs mjpython):

    .venv/bin/mjpython robot_studio/show_model.py                # hexapod
    .venv/bin/mjpython robot_studio/show_model.py --model acrobot.urdf

Tweaks the render look before launching:
  - floor geom -> white
  - background (horizon) -> creamy blue via the visual fog/haze gradient
ESC closes the viewer.
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from robot_studio.model_loader import urdf_to_mjcf, spawn_offset  # noqa: E402

CREAMY_BLUE = (0.78, 0.87, 0.96, 1.0)     # soft blue horizon
FLOOR_WHITE = "1 1 1 1"
FLOOR_NAME = "__ground"


def build(model_file: str):
    ext = model_file.lower().rsplit(".", 1)[-1]
    if ext in ("urdf",):
        xml = urdf_to_mjcf(model_file, add_ground=True)
        # white floor: replace the injected ground plane color
        if FLOOR_NAME in xml:
            # the plane always carries `rgba="0.28 0.31 0.36 0.55"` (see model_loader)
            xml = xml.replace('rgba="0.28 0.31 0.36 0.55"',
                              'rgba="%s"' % FLOOR_WHITE, 1)
    else:  # native MJCF / XML: read as-is
        xml = open(model_file, encoding="utf-8").read()
        # white floor: tame whichever plane geom exists (if any)
        if '<geom' in xml and FLOOR_NAME not in xml and 'type="plane"' in xml:
            xml = xml.replace('type="plane"', 'type="plane" rgba="%s"' % FLOOR_WHITE, 1)
    # creamy-blue horizon: MuJoCo renders the sky as a gradient between
    # vis.rgba.haze (near) and vis.rgba.fog (far distance)
    visual = '<visual><rgba haze="%g %g %g %g" fog="%g %g %g %g"/></visual>' % (
        *CREAMY_BLUE, *CREAMY_BLUE)
    xml = xml.replace("<worldbody>", visual + "<worldbody>", 1)
    return mujoco.MjModel.from_xml_string(xml)


def place_on_floor(model, data):
    """Lift a floating-base robot so its lowest geom just clears the floor.

    The URDF's default rest pose (here: legs straight down) is not a safe
    spawn — the feet end up meters below the ground plane.  We reuse the
    same measurement as ``LeggedEnv`` (``spawn_offset``) and translate the
    free joint so the model starts standing on the ground.
    """
    free = None
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
            free = j
            break
    if free is None:
        return
    info = spawn_offset(model, clearance=0.02)
    idx = model.jnt_qposadr[free]
    data.qpos[idx + 2] += float(info.get("offset", 0.0))
    mujoco.mj_forward(model, data)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="robot_studio/models/pla_hexapod.urdf")
    args = ap.parse_args(argv)
    path = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    if not os.path.isfile(path):
        sys.exit("model not found: %s" % path)
    model = build(path)
    data = mujoco.MjData(model)
    place_on_floor(model, data)
    print("model: %s (nbody=%d ngeom=%d nv=%d nu=%d)" % (
        os.path.basename(path), model.nbody, model.ngeom, model.nv, model.nu))
    print("viewer open - ESC to close")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            v.sync()
    print("closed")


if __name__ == "__main__":
    main()