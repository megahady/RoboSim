"""Terrain / environment composition for robot_studio.

MuJoCo's ``MjModel`` has no ``merge`` in this version (3.10), so the safe way
to give a floating-base robot a real environment is to *splice XML*: take the
MJCF that ``urdf_to_mjcf`` produced, inject the terrain's <asset> and add its
worldbody geoms, then compile once.  This keeps a single model handle, so
``LeggedEnv``, timers, and the viewer all see robot+world as one system.
"""

from __future__ import annotations

import numpy as np
import mujoco

from robot_studio.model_loader import urdf_to_mjcf


def heightmap_fn(nx: int, ny: int, amp: float = 0.2, freq: float = 1.2,
                 seed: int = 0) -> np.ndarray:
    """A deterministic rolling-heightmap function for rough terrain.

    Returns an (ny, nx) grid of heights. ``numpy.random.default_rng(seed)`` is
    used so a terrain is reproducible across runs.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (amp * np.sin(freq * 2 * np.pi * xx / nx)
            + amp * np.cos(freq * 2 * np.pi * yy / ny)
            + 0.3 * amp * rng.standard_normal((ny, nx)))


def hfield_xml(name: str, nx: int, ny: int, sx: float, sy: float,
               hmax: float, z: np.ndarray):
    """Return (asset_snippet, geom_snippet, data) for a procedural hfield.

    In MuJoCo >= 3.x the ``size`` attribute is four numbers ``sx sy hmax
    elevation`` and the last must be positive or compilation fails.  Heights are
    normalized to [0, hmax] so the terrain's lowest point sits at the origin.
    """
    z = np.asarray(z, dtype=float)
    assert z.shape == (ny, nx), "height grid must be (ny, nx)"
    span = float(z.max() - z.min())
    zn = hmax * (z - z.min()) / (span if span > 0 else 1.0)
    return (
        '<hfield name="%s" nrow="%d" ncol="%d" size="%g %g %g 0.001"/>'
        % (name, ny, nx, sx, sy, hmax),
        '<geom name="terrain_%s" type="hfield" hfield="%s" pos="0 0 0" '
        'friction="1.2 0.02 0.001"/>' % (name, name),
        zn.T.ravel(),                     # hfield_data is row-major (nrow, ncol)
    )


def scene_from_urdf(path: str, asset_snippet: str = "", geom_snippet: str = "",
                    hfield_data=None) -> mujoco.MjModel:
    """Compile a URDF robot + optional terrain into a single MjModel.

    ``asset_snippet`` (e.g. ``<hfield .../>``) goes into the model's <asset>
    block (created if the robot has none); ``geom_snippet`` is appended right
    after <worldbody>.  ``hfield_data`` fills in height values for hfields
    declared without a ``file`` (they start zeroed).
    """
    xml = urdf_to_mjcf(path, add_ground=False)
    if asset_snippet:
        if "<asset>" in xml:
            xml = xml.replace("<asset>", "<asset>" + asset_snippet, 1)
        else:
            xml = xml.replace("<worldbody>",
                              "<asset>" + asset_snippet + "</asset><worldbody>", 1)
    if geom_snippet:
        xml = xml.replace("<worldbody>", "<worldbody>" + geom_snippet, 1)
    model = mujoco.MjModel.from_xml_string(xml)
    if hfield_data is not None:
        model.hfield_data[:] = np.asarray(hfield_data).ravel()
    return model


def rough_hexapod(nx: int = 64, ny: int = 64, sx: float = 3.0, sy: float = 3.0,
                  hmax: float = 0.8, seed: int = 0) -> mujoco.MjModel:
    """PLA hexapod on procedurally-generated rough terrain (handy for demo)."""
    asset, geom, data = hfield_xml(
        "rough", nx, ny, sx, sy, hmax, heightmap_fn(nx, ny, seed=seed))
    return scene_from_urdf("robot_studio/models/pla_hexapod.urdf",
                           asset_snippet=asset, geom_snippet=geom,
                           hfield_data=data)