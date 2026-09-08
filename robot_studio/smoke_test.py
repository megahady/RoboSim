"""Headless smoke test: URDF conversion, controller execution, energy values and
optional offscreen 3D scene build.

Run from the HadyLab folder:
    .venv\\Scripts\\python.exe -m robot_studio.smoke_test
    ROBOT_STUDIO_RENDER_TEST=1 .venv\\Scripts\\python.exe -m robot_studio.smoke_test
"""

from __future__ import annotations

import os
import sys

import numpy as np

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def rel(*parts):
    return os.path.join(HERE, *parts)


def main():
    from robot_studio import model_loader
    from robot_studio.controller_manager import ControllerManager

    model_path = rel("models", "acrobot.urdf")
    print("== load URDF ==")
    model = model_loader.load_model(model_path)
    assert model.nu == 2, "expected 2 motors, got %d" % model.nu
    assert model.nv == 2 and model.nq == 2, "expected 2 dofs"
    print(model_loader.model_summary(model))
    for w in model_loader.captured_warnings():
        print("  WARN:", w)

    mgr = ControllerManager(log=lambda s: print("  [log]", s))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    controllers = [
        rel("controllers", "acrobot_inverse_dynamics.py"),
        rel("controllers", "simple_pd_tracking.py"),
    ]
    for cpath in controllers:
        print("== load controller %s ==" % os.path.basename(cpath))
        spec = mgr.load_file(cpath, model, data)
        print("controller:", spec.title, "params:", len(spec.params_def))

        print("== step 400 ==")
        for i in range(400):
            mgr.active().step(model, data)
            if not np.all(np.isfinite(data.qpos)):
                raise RuntimeError("non-finite qpos at step %d" % i)
            mujoco.mj_step(model, data)
        q = np.asarray(data.qpos)
        print("final q    =", np.round(q, 3), "t =", round(float(data.time), 3))
        if not (np.abs(q) < 3.2).all():
            raise RuntimeError("joints outside limits")

        Mf = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, Mf)
        ek = 0.5 * float(np.asarray(data.qvel) @ Mf @ np.asarray(data.qvel))
        print("kinetic energy =", round(ek, 4))
        assert ek < 150.0

        print("== controller reference() ==")
        ref = spec.reference(data, mgr.active().params)
        print("  ref q =", np.round(ref, 3))

    if os.environ.get("ROBOT_STUDIO_RENDER_TEST") == "1":
        print("== offscreen scene build ==")
        import pyvista as pv

        pv.OFF_SCREEN = True
        from robot_studio.scene_viewer import SceneViewer

        plotter = pv.Plotter(off_screen=True)
        viewer = SceneViewer(plotter)
        viewer.load_model(model)
        snap = {
            "xpos": np.array(data.geom_xpos, copy=True),
            "xmat": np.array(data.geom_xmat, copy=True),
            "body_xpos": np.array(data.xpos, copy=True),
            "contacts": {"n": 0, "pos": np.zeros((0, 3)), "force": np.zeros((0, 3)), "frame": np.zeros((0, 3, 3))},
        }
        viewer.apply_state(snap)
        print("  built %d geom actors" % len(viewer.actor_by_geom))

    print("== SMOKE OK ==")


if __name__ == "__main__":
    main()