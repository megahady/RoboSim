"""PyVista rendering of a MuJoCo model: one actor per geom, updated every frame.

MuJoCo publishes per-geom world transforms in ``data.geom_xpos`` / ``data.geom_xmat``.
The viewer rebuilds primitives once at load and then only writes 4x4 ``user_matrix``
per actor, which keeps the render loop cheap even for models with hundreds of geoms.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

import mujoco

_GEOM_META = {
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
    mujoco.mjtGeom.mjGEOM_HFIELD: "hfield",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
}

_MAX_CONTACT_ARROWS = 48

_CAMERA_PRESETS = {
    "orbit": ((2.6, -3.2, 1.9), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "front": ((3.2, 0.0, 0.6), (0.0, 0.0, -0.4), (0.0, 0.0, 1.0)),
    "side": ((0.0, 3.4, 0.7), (0.0, 0.0, -0.4), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0001, 4.2), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
}


class SceneViewer:
    """Builds and animates the PyVista representation of a MuJoCo model."""

    def __init__(self, plotter):
        self.plotter = plotter
        self.model = None
        self.actor_by_geom = {}
        self.contact_arrows = []
        self.show_contacts = False
        self.follow_body = -1
        self.follow_target_com = False
        self._follow_offset = None
        self._up = np.array([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------ build
    def load_model(self, model):
        """Installs a prebuilt mesh list into the plotter (GUI thread)."""
        self.install_model(model, build_geom_data(model))

    def install_model(self, model, items):
        self.model = model
        p = self.plotter
        p.clear()
        self.actor_by_geom = {}
        self._add_ground()
        for i, mesh, rgba in items:
            actor = p.add_mesh(
                mesh,
                color=(rgba[0], rgba[1], rgba[2]),
                opacity=float(rgba[3]),
                name="geom_%d" % i,
                smooth_shading=True,
            )
            self.actor_by_geom[i] = actor
        self._make_contact_arrows()
        p.camera_position = _CAMERA_PRESETS["orbit"]
        p.reset_camera()

    def _add_ground(self):
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=16.0, j_size=16.0)
        grid = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=16.0, j_size=16.0, i_resolution=32, j_resolution=32)
        ground.triangulate()
        self.plotter.add_mesh(ground, color="#23272e", name="_ground", lighting=False)
        self.plotter.add_mesh(grid, color="#2f353d", name="_grid", style="wireframe", line_width=1, lighting=False, opacity=0.6)

    def _make_contact_arrows(self):
        for a in self.contact_arrows:
            self.plotter.remove_actor(a)
        self.contact_arrows = []
        for k in range(_MAX_CONTACT_ARROWS):
            ar = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=1.0,
                          tip_length=0.28, tip_radius=0.08, shaft_radius=0.03)
            actor = self.plotter.add_mesh(ar, color="#ffa726", name="contact_%d" % k, opacity=0.95)
            actor.SetVisibility(0)
            self.contact_arrows.append(actor)
        for a in self.contact_arrows:
            self.plotter.remove_actor(a)
        self.contact_arrows = []
        for k in range(_MAX_CONTACT_ARROWS):
            ar = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=1.0,
                          tip_length=0.28, tip_radius=0.08, shaft_radius=0.03)
            actor = self.plotter.add_mesh(ar, color="#ffa726", name="contact_%d" % k, opacity=0.95)
            actor.SetVisibility(0)
            self.contact_arrows.append(actor)

    # ----------------------------------------------------------------- anim
    def apply_state(self, st):
        if self.model is None or st is None:
            return
        xpos = st["xpos"]
        xmat = st["xmat"]
        for gid, actor in self.actor_by_geom.items():
            T = np.eye(4)
            T[:3, :3] = np.asarray(xmat[gid]).reshape(3, 3)
            T[:3, 3] = xpos[gid]
            actor.user_matrix = T
        if self.show_contacts:
            self._update_contacts(st)
        self._update_follow(st)

    def _update_contacts(self, st):
        data = st.get("contacts")
        n = 0
        if data is not None:
            n = int(data["n"])
            for k in range(min(n, len(self.contact_arrows))):
                pos = data["pos"][k]
                force = data["force"][k]
                mag = float(np.linalg.norm(force))
                if mag < 1e-5:
                    self.contact_arrows[k].SetVisibility(0)
                    continue
                frame = np.asarray(data["frame"][k]).reshape(3, 3)
                normal = frame[0]
                length = float(min(max(mag * 0.12, 0.06), 3.0))
                R = np.column_stack([frame[1], frame[2], normal])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = pos
                S = np.diag([length, length, length, 1.0])
                self.contact_arrows[k].user_matrix = T @ S
                self.contact_arrows[k].SetVisibility(1)
        for k in range(min(n, len(self.contact_arrows)), len(self.contact_arrows)):
            self.contact_arrows[k].SetVisibility(0)

    def _update_follow(self, st):
        if self.follow_body < 0 or self.model is None:
            return
        body_xpos = st.get("body_xpos")
        if body_xpos is None:
            return
        center = body_xpos[self.follow_body]
        if self._follow_offset is None:
            eye = np.asarray(self.plotter.camera_position[0], dtype=float)
            self._follow_offset = eye - center
        self.plotter.camera_position = (center + self._follow_offset, center, self._up)

    # -------------------------------------------------------------- controls
    def set_camera_preset(self, name):
        preset = _CAMERA_PRESETS.get(name)
        if preset:
            self.plotter.camera_position = preset

    def toggle_contacts(self, on: bool):
        self.show_contacts = on
        if not on:
            for a in self.contact_arrows:
                a.SetVisibility(0)

    def enable_follow(self, body_index: int):
        self.follow_body = body_index
        self._follow_offset = None

    def disable_follow(self):
        self.follow_body = -1
        self._follow_offset = None


def _quat_mat(q):
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
            [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
            [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
        ]
    )


def _build_mesh_geom(model, i):
    mid = int(model.geom_dataid[i])
    if mid < 0 or mid >= model.nmesh:
        return None
    vadr = int(model.mesh_vertadr[mid])
    nv = int(model.mesh_vertnum[mid])
    fadr = int(model.mesh_faceadr[mid])
    nf = int(model.mesh_facenum[mid])
    if nv == 0 or nf == 0:
        return None
    verts = np.asarray(model.mesh_vert, dtype=np.float64)[vadr:vadr + nv].reshape(-1, 3).copy()
    faces = np.asarray(model.mesh_face, dtype=np.int64)[fadr:fadr + nf].reshape(-1, 3).copy()
    scale = np.asarray(model.mesh_scale[mid], dtype=np.float64)
    if scale.size == 3:
        verts = verts * scale
    q = np.asarray(model.mesh_quat[mid], dtype=np.float64)
    verts = verts @ _quat_mat(q).T + np.asarray(model.mesh_pos[mid], dtype=np.float64)
    return pv.PolyData(np.ascontiguousarray(verts), np.ascontiguousarray(faces))


def build_geom_data(model):
    """Build the PyVista mesh for every geom (pure data, no VTK widget access).

    Safe to call from a background thread; the caller then hands the result to
    ``SceneViewer.install_model`` on the GUI thread for ``add_mesh`` only.
    """
    items = []
    for i in range(model.ngeom):
        gt = int(model.geom_type[i])
        rgba = np.clip(np.asarray(model.geom_rgba[i], dtype=float), 0.0, 1.0)
        size = model.geom_size[i]
        mesh = None
        if gt == mujoco.mjtGeom.mjGEOM_HFIELD:
            continue
        if gt == mujoco.mjtGeom.mjGEOM_PLANE:
            sx, sy = max(float(size[0]), 1.0), max(float(size[1]), 1.0)
            mesh = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=2 * sx, j_size=2 * sy)
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = pv.Sphere(radius=max(float(size[0]), 1e-4), center=(0, 0, 0), theta_resolution=28, phi_resolution=28)
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = pv.Capsule(center=(0, 0, 0), direction=(0, 0, 1),
                              radius=max(float(size[0]), 1e-4), cylinder_length=2.0 * max(float(size[1]), 1e-6))
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1),
                               radius=max(float(size[0]), 1e-4), height=2.0 * max(float(size[1]), 1e-6), resolution=64)
        elif gt == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            a, b, c = (max(abs(float(size[k])), 1e-4) for k in range(3))
            mesh = pv.Sphere(radius=1.0, theta_resolution=28, phi_resolution=28).scale((a, b, c), inplace=True)
        elif gt == mujoco.mjtGeom.mjGEOM_BOX:
            a, b, c = (max(abs(float(size[k])), 1e-6) for k in range(3))
            mesh = pv.Box(bounds=(-a, a, -b, b, -c, c))
        elif gt == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = _build_mesh_geom(model, i)
        if mesh is not None:
            items.append((i, mesh, rgba))
    return items