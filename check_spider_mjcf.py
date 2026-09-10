import mujoco

MODEL_PATH = r".\robot_studio\models\spider_mjcf.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print(f"loaded: nq={model.nq}, njnt={model.njnt}")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and "femur" in name:
        q0 = data.qpos[model.jnt_qposadr[i]]
        print(f"{name}: q0={q0:.6f} rad, range={model.jnt_range[i]}")
