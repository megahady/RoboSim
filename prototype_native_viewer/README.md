# Native MuJoCo viewer prototype

A **standalone, throwaway** comparison of MuJoCo's native OpenGL viewer
(`mujoco.viewer`) against the app's current PyVista strategy
(`robot_studio/scene_viewer.py`). It does **not** touch the working app — it
reuses only the shared `model_loader` and `controller_manager`.

Not committed to `requirements.txt`; nothing to install.

## Run

```bash
# Classic: our own physics+controller thread, native viewer synced to it (macOS: under mjpython)
.venv/bin/mjpython prototype_native_viewer/native_viewer.py
.venv/bin/mjpython prototype_native_viewer/native_viewer.py \
    robot_studio/models/pla_hexapod.urdf \
    robot_studio/controllers/hexapod_tripod_gait.py

# Simulate: MuJoCo's built-in loop (plain python is fine; no controller hook)
.venv/bin/python prototype_native_viewer/native_viewer.py --mode simulate
```

## Modes

| mode | function | physics/controller | threading | macOs |
|------|----------|--------------------|-----------|-------|
| `classic` (default) | `launch_passive` | ours (ControllerManager, 4 substeps/iter) | separate physics loop + viewer sync | needs `mjpython` |
| `simulate` | `launch` | MuJoCo built-in (no controller hook) | one loop | plain python |

## Key findings (so far)

- Native viewer renders the model **directly from `MjModel`/`MjData`** — no
  mesh rebuild in Python, no per-geom VTK actor bookkeeping, no manual
  `user_matrix` uploads. Visually matches MuJoCo exactly (meshes, textures,
  lighting). Much less code than `scene_viewer.py:209` `build_geom_data`.
- **macOS needs `mjpython`** for the passive viewer (`WinError 1114` on
  Windows / framework requirement here). This is the single biggest integration
  friction.
- `--mode simulate` gives you a full native window but **no hook to run your
  controllers** — that's a deal-breaker for this tool's whole point (tuning
  your own control scripts).
- `--mode classic` runs your controllers but ties the viewer's `sync()` into
  our physics loop: a slow viewer frame now stalls physics, exactly what the
  README's "Critical Mistakes #2" warns against.

## Verdict this prototype is meant to inform

Native viewer = less rendering code + exact fidelity, but brings two costs
relative to the current PyVista design: framework/`mjpython` portability and
the loss of the clean physics/viewer thread separation (plus custom visuals
like follow-camera-with-offset and force arrows that aren't free in the native
viewer). See `../README.md` "3D view strategy" discussion.
