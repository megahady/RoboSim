"""Model loading for robot_studio.

Supports native MJCF/XML and URDF. URDF models are converted to MJCF and every
revolute/continuous/prismatic joint gets a torque motor injected, so that any
loaded robot is immediately controllable through ``data.ctrl``. Floating base
support: a URDF ``<joint type="floating">`` becomes a MuJoCo ``<freejoint/>``.
"""

from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import mujoco

WORLD_LINK = "world"


class ModelLoadError(Exception):
    pass


_warnings: list[str] = []


def captured_warnings() -> list[str]:
    return list(_warnings)


def _warn(msg: str) -> None:
    _warnings.append(msg)


def _fmt(vals) -> str:
    return " ".join("%.6g" % float(x) for x in vals)


def _vec(el, attr, default):
    s = el.get(attr) if el is not None else None
    if not s:
        return list(default)
    try:
        return [float(x) for x in s.replace(",", " ").split()]
    except ValueError as exc:
        raise ModelLoadError("Bad vector %r = %r" % (attr, s)) from exc


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _quat_euler(roll, pitch, yaw):
    qx = [math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0]
    qy = [math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0]
    qz = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    return _qmul(qz, _qmul(qy, qx))


def _rot_inertia(inertia, rpy):
    """Rotate a URDF inertia tensor (ixx,ixy,ixz,iyy,iyz,izz) by fixed-axis ZYX rpy."""
    ixx, ixy, ixz, iyy, iyz, izz = (float(x) for x in inertia)
    I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    roll, pitch, yaw = (float(x) for x in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    I2 = R @ I @ R.T
    return [I2[0, 0], I2[0, 1], I2[0, 2], I2[1, 1], I2[1, 2], I2[2, 2]]


def _sanitize_inertia(inertia):
    """Clamp an inertia tensor to a physically valid one (positive eigenvalues
    and triangle inequality) so that sloppy URDF inertias still load."""
    ixx, ixy, ixz, iyy, iyz, izz = (float(x) for x in inertia)
    M = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    M = (M + M.T) / 2.0
    evals, evecs = np.linalg.eigh(M)
    floor = max(abs(evals).max() * 1e-4, 1e-6)
    evals = np.clip(evals, floor, None)
    order = np.argsort(evals)[::-1]
    a, b, c = evals[order]
    if a >= b + c:
        a = b + c + max(floor, abs(a) * 1e-2)
    evals[order] = np.array([a, b, c])
    M2 = evecs @ np.diag(evals) @ evecs.T
    return [M2[0, 0], M2[0, 1], M2[0, 2], M2[1, 1], M2[1, 2], M2[2, 2]]


def _materials(root):
    mats = {}
    for m in root.findall("material"):
        c = m.find("color")
        if c is not None and c.get("rgba"):
            try:
                mats[m.get("name")] = [float(x) for x in c.get("rgba").split()]
            except ValueError:
                pass
    return mats


def _parse_joints(root):
    joints = []
    for el in root.findall("joint"):
        typ = (el.get("type") or "continuous").lower()
        origin = el.find("origin")
        ax = el.find("axis")
        lim = el.find("limit")
        lower, upper, effort = -6.28, 6.28, 100.0
        if typ in ("revolute", "prismatic") and lim is not None:
            if lim.get("lower") is not None:
                lower = float(lim.get("lower"))
            if lim.get("upper") is not None:
                upper = float(lim.get("upper"))
        if lim is not None and lim.get("effort"):
            effort = float(lim.get("effort"))
        parent_el = el.find("parent")
        child_el = el.find("child")
        if parent_el is None or child_el is None:
            continue
        joints.append(
            {
                "name": el.get("name") or ("j%d" % len(joints)),
                "type": typ,
                "parent": parent_el.get("link"),
                "child": child_el.get("link"),
                "origin": _vec(origin, "xyz", (0, 0, 0)),
                "rpy": _vec(origin, "rpy", (0, 0, 0)),
                "axis": _vec(ax, "xyz", (0, 0, 1)),
                "lower": lower,
                "upper": upper,
                "effort": effort,
            }
        )
    return joints


def _parse_links(root, mats):
    links = {}
    for el in root.findall("link"):
        name = el.get("name")
        inertial = el.find("inertial")
        mass = None
        inertia = [0.0] * 6
        ipos = (0, 0, 0)
        irpy = (0, 0, 0)
        if inertial is not None:
            mi = inertial.find("mass")
            ii = inertial.find("inertia")
            io = inertial.find("origin")
            if mi is not None:
                mass = float(mi.get("value", 0.0))
            if ii is not None:
                inertia = [
                    float(ii.get(k, 0.0))
                    for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
                ]
            if io is not None:
                ipos = _vec(io, "xyz", (0, 0, 0))
                irpy = _vec(io, "rpy", (0, 0, 0))
        geoms = []
        for vis in el.findall("visual"):
            geo = vis.find("geometry")
            if geo is None:
                continue
            found = None
            for tag in ("box", "cylinder", "sphere", "capsule", "mesh"):
                sub = geo.find(tag)
                if sub is not None:
                    found = {"kind": tag, "attr": dict(sub.attrib)}
                    break
            if found is None:
                continue
            found["origin"] = _vec(vis.find("origin"), "xyz", (0, 0, 0))
            found["rpy"] = _vec(vis.find("origin"), "rpy", (0, 0, 0))
            mat = vis.find("material")
            rgba = [0.62, 0.62, 0.62, 1.0]
            if mat is not None:
                if mat.get("rgba"):
                    try:
                        rgba = [float(x) for x in mat.get("rgba").split()]
                    except ValueError:
                        pass
                elif mat.get("name") in mats:
                    rgba = mats[mat.get("name")]
            found["rgba"] = rgba
            geoms.append(found)
        links[name] = {
            "name": name,
            "mass": mass,
            "inertia": inertia,
            "ipos": ipos,
            "irpy": irpy,
            "geoms": geoms,
        }
    return links


def _mesh_filename(raw: str) -> str:
    for prefix in ("package://", "model://", "file://"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    return raw.replace("\\", "/").split("/")[-1]


def _rect_mesh_geom(g, base, registrar):
    raw = g["attr"].get("filename", "")
    if not raw:
        return None
    fname = _mesh_filename(raw)
    resolved = os.path.join(base, fname)
    if not os.path.isfile(resolved):
        _warn("Mesh file not found: %s (skipped)" % resolved)
        return None
    scale_raw = g["attr"].get("scale", "1 1 1")
    try:
        scale = [float(x) for x in scale_raw.split()]
    except ValueError:
        scale = [1.0, 1.0, 1.0]
    while len(scale) < 3:
        scale.append(1.0)
    mname = registrar(resolved, scale[:3])
    quat = _quat_euler(*g["rpy"])
    return '<geom type="mesh" mesh="%s" pos="%s" quat="%s" rgba="%s"/>' % (
        mname,
        _fmt(g["origin"]),
        _fmt(quat),
        ",".join("%g" % x for x in g["rgba"]),
    )


_CAPSULE_KEYWORDS = ("radius", "length", "cylinder_radius", "cylinder_length")


def _rect_geom_xml(g, base, registrar):
    kind = g["kind"]
    attr = g["attr"]
    quat = _quat_euler(*g["rpy"])
    rgba = " ".join("%g" % x for x in g["rgba"])
    pos = _fmt(g["origin"])
    qs = _fmt(quat)
    if kind == "box":
        try:
            size = [float(x) / 2.0 for x in attr["size"].split()]
        except Exception as exc:
            raise ModelLoadError("Bad box size in URDF") from exc
        return '<geom type="box" pos="%s" quat="%s" size="%s" rgba="%s"/>' % (pos, qs, _fmt(size), rgba)
    if kind == "cylinder":
        return '<geom type="cylinder" pos="%s" quat="%s" size="%s %s" rgba="%s"/>' % (
            pos, qs, attr.get("radius", "0.02"), float(attr.get("length", "0.1")) / 2.0, rgba
        )
    if kind == "sphere":
        return '<geom type="sphere" pos="%s" quat="%s" size="%s" rgba="%s"/>' % (
            pos, qs, attr.get("radius", "0.02"), rgba
        )
    if kind == "capsule":
        r = attr.get("radius") or attr.get("cylinder_radius") or "0.02"
        l = attr.get("length") or attr.get("cylinder_length") or "0.1"
        return '<geom type="capsule" pos="%s" quat="%s" size="%s %s" rgba="%s"/>' % (
            pos, qs, r, float(l) / 2.0, rgba
        )
    if kind == "mesh":
        return _rect_mesh_geom(g, base, registrar)
    return None


def urdf_to_mjcf(path: str, timestep: float = 0.005, damping: float = 0.05, armature: float = 0.01,
                 add_ground: bool | None = None) -> str:
    """Convert a URDF file into an MJCF XML string with motors on every joint.

    ``add_ground``: ``None`` (default) adds a ground plane automatically for
    free-floating (mobile) robots, ``True`` forces one, ``False`` never.
    """
    global _warnings
    _warnings = []
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ModelLoadError("Not a URDF <robot> document: %s" % path)
    base = os.path.dirname(os.path.abspath(path))
    name = root.get("name") or os.path.splitext(os.path.basename(path))[0]
    mats = _materials(root)
    links = _parse_links(root, mats)
    joints = _parse_joints(root)

    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    roots = [n for n in links if not any(j["child"] == n for j in joints)]
    world_known = WORLD_LINK in links or any(j["parent"] == WORLD_LINK for j in joints)
    floating = (
        not world_known
        and len(roots) == 1
        and roots[0] != WORLD_LINK
        and bool(links[roots[0]]["mass"])
    )
    for r in roots:
        if r not in children and r != WORLD_LINK:
            _warn("Root link %r has no joint; it is treated as a %s base."
                  % (r, "free-floating" if floating else "welded/tied to the world"))
    if not roots:
        raise ModelLoadError("No root link found in URDF")

    if WORLD_LINK in roots:
        root_link = WORLD_LINK
    else:
        root_link = roots[0]

    meshes = []
    registrar = {"i": 0}
    mesh_names = []

    def register_mesh(filepath, scale):
        mid = registrar["i"]
        registrar["i"] += 1
        mname = "mesh_%d" % mid
        mesh_names.append(mname)
        meshes.append('<mesh name="%s" file="%s" scale="%s"/>' % (mname, os.path.basename(filepath), _fmt(scale)))
        return mname

    out = ['<mujoco model="%s">' % name]
    out.append('<compiler angle="radian" coordinate="local" meshdir="%s"/>' % base)
    out.append('<option timestep="%s" integrator="Euler"/>' % timestep)
    if damping > 0.0 or armature > 0.0:
        out.append("<default>")
        out.append('  <joint damping="%s" armature="%s"/>' % (damping, armature))
        out.append("</default>")
    if meshes:
        out.append("<asset>")
        out.extend("  " + m for m in meshes)
        out.append("</asset>")

    seen = set()
    body_buf = []

    def emit(link_name, joint, depth):
        pad = "  " * depth
        lk = links.get(link_name)
        if lk is None:
            if joint is None:
                return
            lk = {"name": link_name, "mass": None, "inertia": [0.0] * 6,
                  "ipos": (0, 0, 0), "irpy": (0, 0, 0), "geoms": []}
        if link_name == WORLD_LINK:
            for g in lk["geoms"]:
                gx = _rect_geom_xml(g, base, register_mesh)
                if gx:
                    body_buf.append(pad + gx)
        else:
            if link_name in seen:
                return
            seen.add(link_name)
            pos = joint["origin"] if joint else (0, 0, 0)
            quat = _quat_euler(*joint["rpy"]) if joint else (1, 0, 0, 0)
            jtyp = joint["type"] if joint else "fixed"
            head = None
            free = jtyp == "floating" or (floating and joint is None and link_name == root_link)
            if free:
                head = '<body name="%s">' % link_name
            else:
                head = '<body name="%s" pos="%s" quat="%s">' % (link_name, _fmt(pos), _fmt(quat))
            body_buf.append(pad + head)
            if free:
                body_buf.append(pad + "  " + '<freejoint name="%s_free"/>' % link_name)
            elif joint is not None and jtyp not in ("fixed",):
                if jtyp in ("revolute", "continuous"):
                    jtag = "hinge"
                    rng = "" if jtyp == "continuous" else ' range="%s %s"' % (_fmt([joint["lower"]]), _fmt([joint["upper"]]))
                elif jtyp == "prismatic":
                    jtag = "slide"
                    rng = ' range="%s %s"' % (_fmt([joint["lower"]]), _fmt([joint["upper"]]))
                else:
                    jtag = "hinge"
                    rng = ""
                    _warn("Unsupported joint type %r mapped to hinge." % jtyp)
                body_buf.append(pad + "  " + '<joint name="%s" type="%s" axis="%s"%s/>' % (
                    joint["name"], jtag, _fmt(joint["axis"]), rng))
            if lk["mass"]:
                full = _sanitize_inertia(_rot_inertia(lk["inertia"], lk["irpy"]))
                full = [full[0], full[3], full[5], full[1], full[2], full[4]]
                body_buf.append(pad + "  " + '<inertial pos="%s" mass="%s" fullinertia="%s"/>' % (
                    _fmt(lk["ipos"]), lk["mass"], _fmt(full)))
            for g in lk["geoms"]:
                gx = _rect_geom_xml(g, base, register_mesh)
                if gx:
                    body_buf.append(pad + "  " + gx)
            for j in children.get(link_name, []):
                emit(j["child"], j, depth + 1)
            body_buf.append(pad + "</body>")

    out.append("<worldbody>")
    if add_ground if add_ground is not None else floating:
        out.append('  <geom name="__ground" type="plane" size="6 6 0.02" pos="0 0 0" '
                   'rgba="0.28 0.31 0.36 0.55" friction="1.2 0.02 0.001"/>')
    emit(root_link, None, 1)
    for j in children.get(root_link, []):
        emit(j["child"], j, 1)
    out.extend(body_buf)
    out.append("</worldbody>")

    motors = []
    for j in joints:
        if j["type"] in ("revolute", "continuous", "prismatic"):
            effort = abs(j["effort"]) or 100.0
            motors.append('  <motor joint="%s" name="act_%s" ctrlrange="%s %s" ctrllimited="true"/>' % (
                j["name"], j["name"], -effort, effort))
    if motors:
        out.append("<actuator>")
        out.extend(motors)
        out.append("</actuator>")

    out.append("</mujoco>")
    return "\n".join(out) + "\n"


def load_model_urdf(path: str) -> mujoco.MjModel:
    xml_text = urdf_to_mjcf(path)
    fd, tmp = tempfile.mkstemp(suffix=".xml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(xml_text)
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def spawn_offset(model: mujoco.MjModel, clearance: float = 0.02) -> dict:
    """Measure the lowest body point of the model in its rest pose and return
    the z-lift (+ clearance hover) needed so a floating-base robot starts just
    above the floor instead of interpenetrating it.

    Returns ``{"free": bool, "lowest": float, "offset": float}`` where
    ``offset`` is the free-joint z translation to apply to ``data.qpos``.
    ``free`` is False (offset 0) for fixed-base models.
    """
    free = False
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
            free = True
            break
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    lowest = float("inf")
    for g in range(model.ngeom):
        gt = int(model.geom_type[g])
        if gt in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
            continue
        s = np.asarray(model.geom_size[g], dtype=float)
        r = float(np.sqrt(float(np.dot(s, s))))
        lowest = min(lowest, float(d.geom_xpos[g, 2]) - r)
    if not free or not np.isfinite(lowest) or lowest >= clearance:
        return {"free": bool(free), "lowest": lowest, "offset": 0.0}
    return {"free": True, "lowest": lowest, "offset": clearance - lowest}


def load_model(path: str) -> mujoco.MjModel:
    """Load a MJCF/XML or URDF model. Returns a compiled mujoco.MjModel."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ModelLoadError("Model file not found: %s" % path)
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "xml":
        return mujoco.MjModel.from_xml_path(path)
    if ext == "urdf":
        return load_model_urdf(path)
    raise ModelLoadError("Unsupported model extension: .%s (use .urdf or .xml)" % ext)


def name_of(model: mujoco.MjModel, obj_type, index):
    """Item name via mujoco.mj_id2name (returns None when unnamed)."""
    return mujoco.mj_id2name(model, obj_type, index)


def model_summary(model: mujoco.MjModel) -> str:
    motors = [name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)] if model.nu else []
    mname = ""
    if hasattr(model, "names"):
        mname = bytes(model.names).split(b"\x00", 1)[0].decode(errors="replace")
    return (
        "model: %s | bodies: %d | geoms: %d | joints: %d | dofs: %d | actuators: %d (%s)"
        % (
            mname or "?",
            model.nbody,
            model.ngeom,
            model.njnt,
            model.nv,
            model.nu,
            ", ".join(motors) if motors else "none",
        )
    )