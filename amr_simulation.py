"""
AMR (Autonomous Mobile Robot) MuJoCo Simulation
Differential-drive robot navigating from (0,0) to (4,4).

Architecture:
  - Three planar DOFs (slide-X, slide-Y, hinge-Yaw) for stable body control
  - Moderate-gain velocity servos drive the body directly in world frame
  - Two wheel hinges animate the spinning wheels visually
  - All collisions disabled; floor is purely visual

Run:  python amr_simulation.py
"""

import numpy as np
import mujoco
import mujoco.viewer

# ── XML model ──────────────────────────────────────────────────────────────────
MODEL_XML = """
<mujoco model="amr">
  <option gravity="0 0 -9.81" timestep="0.005" integrator="RK4"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <rgba haze="0.8 0.9 1.0 1"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             width="512" height="512"
             rgb1="0.85 0.85 0.90" rgb2="0.70 0.70 0.75"/>
    <material name="floor_mat" texture="grid" texrepeat="6 6" reflectance="0.15"/>
    <material name="body_mat"  rgba="0.20 0.55 0.85 1"/>
    <material name="wheel_mat" rgba="0.15 0.15 0.15 1"/>
    <material name="goal_mat"  rgba="0.10 0.85 0.30 0.55"/>
    <material name="trail_mat" rgba="1.00 0.65 0.10 0.80"/>
  </asset>

  <worldbody>
    <!-- Floor - visual only, no collision -->
    <geom name="floor" type="plane" size="8 8 0.1"
          material="floor_mat" contype="0" conaffinity="0" pos="2 2 0"/>

    <!-- Goal marker -->
    <geom name="goal" type="cylinder" size="0.18 0.01"
          pos="4 4 0.01" material="goal_mat"
          contype="0" conaffinity="0"/>

    <!-- Start marker -->
    <geom name="start" type="cylinder" size="0.12 0.01"
          pos="0 0 0.01" material="trail_mat"
          contype="0" conaffinity="0"/>

    <!-- AMR body - three free planar DOFs -->
    <body name="base" pos="0 0 0.12">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="5"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="5"/>
      <joint name="hinge_yaw" type="hinge" axis="0 0 1" damping="3"/>

      <!-- Main chassis -->
      <geom name="chassis" type="box" size="0.22 0.16 0.07"
            material="body_mat" contype="0" conaffinity="0"/>

      <!-- Sensor bump on top -->
      <geom name="sensor_top" type="cylinder" size="0.06 0.04"
            pos="0.10 0 0.07" material="body_mat"
            contype="0" conaffinity="0"/>

      <!-- Left wheel – damping < I/dt ≈ 0.49 to stay numerically stable -->
      <body name="wheel_left" pos="-0.05 0.18 -0.05">
        <joint name="wheel_left_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom name="wl" type="cylinder" size="0.07 0.04"
              material="wheel_mat" contype="0" conaffinity="0"
              euler="90 0 0"/>
      </body>

      <!-- Right wheel -->
      <body name="wheel_right" pos="-0.05 -0.18 -0.05">
        <joint name="wheel_right_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom name="wr" type="cylinder" size="0.07 0.04"
              material="wheel_mat" contype="0" conaffinity="0"
              euler="90 0 0"/>
      </body>

      <!-- Front caster (visual) -->
      <geom name="caster" type="sphere" size="0.04"
            pos="0.18 0 -0.05" material="wheel_mat"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Body velocity servos (world frame) only – wheels animated via qvel -->
    <velocity name="act_x"   joint="slide_x"   kv="120"/>
    <velocity name="act_y"   joint="slide_y"   kv="120"/>
    <velocity name="act_yaw" joint="hinge_yaw" kv="80"/>
  </actuator>
</mujoco>
"""

# ── Controller ─────────────────────────────────────────────────────────────────
GOAL      = np.array([4.0, 4.0])
MAX_VEL   = 0.01         # m/s  – moderate, intentionally not fast
MAX_YAW   = 1.2           # rad/s
K_P_POS   = 1.0
K_P_YAW   = 2.5
ARRIVE_R  = 0.12          # metres
WHEEL_R   = 0.07          # wheel radius for spin animation


def get_pos(data) -> np.ndarray:
    return data.qpos[0:2].copy()


def get_yaw(data) -> float:
    return float(data.qpos[2])


def angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_controls(data):
    """Returns (ctrl_3, wheel_spin_rad_s) for body servos + visual wheel rate."""
    pos = get_pos(data)
    yaw = get_yaw(data)
    err = GOAL - pos
    dist = np.linalg.norm(err)

    if dist < ARRIVE_R:
        return np.zeros(3), 0.0

    desired_yaw = np.arctan2(err[1], err[0])
    yaw_err     = angle_wrap(desired_yaw - yaw)

    speed  = np.clip(K_P_POS * dist, 0.0, MAX_VEL)
    vx     = speed * np.cos(desired_yaw)
    vy     = speed * np.sin(desired_yaw)
    v_yaw  = np.clip(K_P_YAW * yaw_err, -MAX_YAW, MAX_YAW)

    wheel_spin = np.linalg.norm([vx, vy]) / WHEEL_R
    return np.array([vx, vy, v_yaw]), wheel_spin


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data  = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    arrived   = False
    hold_time = 0.0
    HOLD_SECS = 1.5      # stay at goal a moment before exit

    print("AMR simulation starting ...")
    print(f"  Target : {GOAL}")
    print(f"  Max vel: {MAX_VEL} m/s  (moderate)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Nice camera angle
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0]
        viewer.cam.distance   = 9.5
        viewer.cam.elevation  = -35
        viewer.cam.azimuth    = 45

        while viewer.is_running():
            pos  = get_pos(data)
            dist = np.linalg.norm(GOAL - pos)

            if not arrived:
                ctrl, wheel_spin = compute_controls(data)
                data.ctrl[:] = ctrl
                # Inject wheel qvel directly – avoids actuator instability on tiny inertia
                data.qvel[3] =  wheel_spin   # left  wheel (rolls forward)
                data.qvel[4] = -wheel_spin   # right wheel (rolls forward)
                if dist < ARRIVE_R:
                    arrived = True
                    data.ctrl[:] = 0.0
                    print(f"  Arrived at goal!  pos={pos}  t={data.time:.2f}s")
            else:
                # Wheels must be explicitly zeroed – physics alone overshoots
                data.qvel[3] = 0.0
                data.qvel[4] = 0.0
                hold_time += model.opt.timestep
                if hold_time > HOLD_SECS:
                    print("  Done. Closing viewer.")
                    break

            mujoco.mj_step(model, data)
            viewer.sync()

    print("Simulation finished.")


if __name__ == "__main__":
    main()
