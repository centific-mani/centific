"""
Integrated Warehouse Simulation — Three-Stage Pipeline
  Stage 1 : Franka Panda #1 + simulated YOLOv8  → picks package by priority
  Stage 2 : AMR + LiDAR A* replanning           → delivers package to truck bay
  Stage 3 : Franka Panda #2 + LIFO/BLB/RRT     → loads package into truck

All three stages run in a single MuJoCo world.
Actuator layout:
  ctrl[0-7]   Panda #1  (actuator1 … actuator8)
  ctrl[8-10]  AMR       (act_x, act_y, act_yaw)
  ctrl[11-18] Panda #2  (s3_actuator1 … s3_actuator8)

Run:  python integrated_warehouse.py
"""

import numpy as np
import mujoco
import mujoco.viewer
from enum import Enum, auto

from stage1_pick_station  import PickStation
from stage3_truck_loading import TruckLoadingStation
from amr_payload          import PayloadManager
from amr_simulation import (
    astar, smooth_path, scan_lidar, draw_lidar,
    update_grid_from_lidar, angle_wrap,
    LIDAR_RAYS, LIDAR_RANGE, _LIDAR_ANGLES,
    GRID_COLS, GRID_ROWS,
    MAX_VEL, MAX_YAW, K_P_POS, K_P_YAW, ARRIVE_R, WAYPOINT_R, WHEEL_R, SAFETY_R,
)

# ── Layout constants ───────────────────────────────────────────────────────────
ARM_BASE      = (0.3,  0.0,  0.0)
CONV_OFFSET   = (0.0, -0.45, 0.65)
LOAD_ZONE     = np.array([0.45, 0.50])
DELIVERY_GOAL = np.array([7.0,  7.0])
AMR_START     = np.array([0.0,  0.0])

# ── Pipeline phases ────────────────────────────────────────────────────────────
class Phase(Enum):
    DRIVE_TO_LOAD = auto()
    WAIT_FOR_PKG  = auto()
    LOAD_PAYLOAD  = auto()
    NAVIGATE      = auto()
    STAGE3_UNLOAD = auto()
    COMPLETE      = auto()

# ── AMR robot XML fragments ────────────────────────────────────────────────────
AMR_WORLDBODY_XML = """\
    <body name="base" pos="0 0 0.18">
      <inertial pos="0 0 0" mass="20" diaginertia="0.45 0.81 1.10"/>
      <joint name="slide_x"   type="slide" axis="1 0 0" damping="5"/>
      <joint name="slide_y"   type="slide" axis="0 1 0" damping="5"/>
      <joint name="hinge_yaw" type="hinge" axis="0 0 1" damping="3"/>
      <camera name="depth_cam" pos="0.33 0 0.07" xyaxes="0 -1 0 0 0 1"/>
      <geom name="chassis"    type="box"      size="0.33 0.24 0.10"
            material="body_mat" contype="0" conaffinity="0" group="1"/>
      <geom name="sensor_top" type="cylinder" size="0.09 0.06" pos="0.15 0 0.10"
            material="body_mat" contype="0" conaffinity="0" group="1"/>
      <body name="cargo_body" pos="0 0 0.22">
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
        <geom name="cargo" type="box" size="0.24 0.18 0.12"
              rgba="0 0 0 0" contype="0" conaffinity="0" group="1"/>
      </body>
      <body name="wheel_left"  pos="-0.07  0.27 -0.07">
        <joint name="wheel_left_spin"  type="hinge" axis="0 1 0" damping="0.3"/>
        <geom type="cylinder" size="0.10 0.06"
              material="wheel_mat" contype="0" conaffinity="0" euler="1.5708 0 0" group="1"/>
      </body>
      <body name="wheel_right" pos="-0.07 -0.27 -0.07">
        <joint name="wheel_right_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom type="cylinder" size="0.10 0.06"
              material="wheel_mat" contype="0" conaffinity="0" euler="1.5708 0 0" group="1"/>
      </body>
      <geom name="caster" type="sphere" size="0.06" pos="0.27 0 -0.07"
            material="wheel_mat" contype="0" conaffinity="0" group="1"/>
    </body>"""

AMR_ACTUATOR_XML = """\
    <velocity name="act_x"   joint="slide_x"   kv="120"/>
    <velocity name="act_y"   joint="slide_y"   kv="120"/>
    <velocity name="act_yaw" joint="hinge_yaw" kv="80"/>"""


# ── AMR controller ─────────────────────────────────────────────────────────────
class AMRController:
    def setup(self, model, data):
        def jqp(n): return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        def jqv(n): return model.jnt_dofadr [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        def aid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

        self.sx    = jqp("slide_x");  self.sy   = jqp("slide_y");  self.syaw = jqp("hinge_yaw")
        self.wlv   = jqv("wheel_left_spin");   self.wrv = jqv("wheel_right_spin")
        self.ax    = aid("act_x");    self.ay   = aid("act_y");    self.ayaw = aid("act_yaw")
        self.cargo_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cargo_body")
        self.cargo_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cargo")

    def pos(self, data):  return np.array([data.qpos[self.sx], data.qpos[self.sy]])
    def yaw(self, data):  return float(data.qpos[self.syaw])

    def drive(self, data, waypoint, ranges) -> bool:
        pos  = self.pos(data)
        err  = np.array(waypoint) - pos
        dist = np.linalg.norm(err)
        if dist < 1e-3:
            self._zero(data); return True

        dyaw  = np.arctan2(err[1], err[0])
        speed = np.clip(K_P_POS * dist, 0.0, MAX_VEL)

        min_i = int(np.argmin(ranges)); mr = float(ranges[min_i])
        if mr < SAFETY_R:
            oa  = float(_LIDAR_ANGLES[min_i])
            rep = np.array([-np.cos(oa), -np.sin(oa)])
            gd  = err / (dist + 1e-9)
            w   = 1.0 - mr / SAFETY_R
            bl  = rep * w + gd * (1.0 - w * 0.5)
            bn  = np.linalg.norm(bl)
            if bn > 1e-6: bl /= bn
            dyaw  = np.arctan2(bl[1], bl[0])
            speed = np.clip(K_P_POS * dist, 0.0, MAX_VEL * (mr / SAFETY_R))

        yerr = angle_wrap(dyaw - self.yaw(data))
        vx   = speed * np.cos(dyaw); vy = speed * np.sin(dyaw)
        vyaw = np.clip(K_P_YAW * yerr, -MAX_YAW, MAX_YAW)
        ws   = np.linalg.norm([vx, vy]) / WHEEL_R
        data.ctrl[self.ax] = vx; data.ctrl[self.ay] = vy; data.ctrl[self.ayaw] = vyaw
        data.qvel[self.wlv] = ws; data.qvel[self.wrv] = ws
        return dist < ARRIVE_R

    def _zero(self, data):
        data.ctrl[self.ax] = data.ctrl[self.ay] = data.ctrl[self.ayaw] = 0.0
        data.qvel[self.wlv] = data.qvel[self.wrv] = 0.0

    def stop(self, data): self._zero(data)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # ── Stage 3: get XML snippets first (needed for combined XML build) ────────
    stage3 = TruckLoadingStation()
    s3     = stage3.get_xml_snippets()

    # ── Build combined scene XML with all three stages ─────────────────────────
    ps = PickStation(base_pos=ARM_BASE, conveyor_offset=CONV_OFFSET)
    xml_path = ps.build_combined_xml(
        amr_worldbody_xml    = AMR_WORLDBODY_XML + "\n" + s3["worldbody"],
        amr_actuator_xml     = AMR_ACTUATOR_XML  + "\n    " + s3["actuator"],
        extra_tendon_inner   = s3["tendon"],
        extra_equality_inner = s3["equality"],
        extra_contact_inner  = s3["contact"],
        load_zone            = tuple(LOAD_ZONE),
    )
    print(f"[Integrated] Combined scene XML: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Place goal marker
    gm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
    data.mocap_pos[model.body_mocapid[gm_bid]] = [DELIVERY_GOAL[0], DELIVERY_GOAL[1], 0.012]

    # ── Setup all three stage controllers ──────────────────────────────────────
    ps.setup(model, data)
    amr     = AMRController(); amr.setup(model, data)
    payload = PayloadManager(robot_name="AMR-01")
    stage3.setup(model, data)

    print(f"\n  Panda #1 ctrl[0..7]  act_x={amr.ax}  act_y={amr.ay}  act_yaw={amr.ayaw}")

    grid         = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    total_hits   = 0
    replan_count = 0
    ranges       = np.full(LIDAR_RAYS, LIDAR_RANGE)

    raw       = astar(grid, tuple(AMR_START), tuple(LOAD_ZONE))
    waypoints = smooth_path(raw) if raw else [tuple(LOAD_ZONE)]
    wp_idx    = min(1, len(waypoints) - 1)

    phase        = Phase.DRIVE_TO_LOAD
    pending_pkg  = None
    hold_ctr     = 0
    step         = 0

    print("\n" + "="*66)
    print("  INTEGRATED WAREHOUSE — Stage 1 → Stage 2 → Stage 3")
    print("="*66)
    print(f"  Franka #1 base : {ARM_BASE}")
    print(f"  Load zone      : {tuple(LOAD_ZONE)}")
    print(f"  Delivery goal  : {tuple(DELIVERY_GOAL)}")
    print(f"  Franka #2 base : (7.5, 7.0, 0.0)")
    print(f"  Staging table  : (7.5, 6.3)  h=0.65m")
    print(f"  Truck          : (8.3, 7.0)")
    print("="*66 + "\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [4.0, 3.5, 0.5]
        viewer.cam.distance  = 22.0
        viewer.cam.elevation = -28
        viewer.cam.azimuth   = 50

        while viewer.is_running():
            pos = amr.pos(data)

            # ── LiDAR ──────────────────────────────────────────────────────────
            ranges = scan_lidar(model, data, pos)

            # ── Grid update + replan ───────────────────────────────────────────
            if phase in (Phase.DRIVE_TO_LOAD, Phase.NAVIGATE):
                if update_grid_from_lidar(grid, pos, ranges):
                    new_hits   = int(grid.sum()) - total_hits
                    total_hits = int(grid.sum())
                    goal_now   = LOAD_ZONE if phase == Phase.DRIVE_TO_LOAD \
                                 else DELIVERY_GOAL
                    raw = astar(grid, tuple(pos), tuple(goal_now))
                    if raw:
                        waypoints    = smooth_path(raw)
                        wp_idx       = min(1, len(waypoints) - 1)
                        replan_count += 1
                        print(f"  [replan #{replan_count}  t={data.time:.1f}s]  "
                              f"+{new_hits} obstacle cells  "
                              f"{len(waypoints)} waypoints")

            # ── Stage 1 arm tick (runs every step) ────────────────────────────
            arm_result = ps.step(model, data, step)
            if arm_result is not None and phase == Phase.WAIT_FOR_PKG:
                pending_pkg = arm_result
                phase = Phase.LOAD_PAYLOAD

            # ── Stage 3 arm tick (runs every step during unload) ──────────────
            if phase == Phase.STAGE3_UNLOAD:
                s3_done = stage3.step(model, data, step)
                if s3_done:
                    phase = Phase.COMPLETE
                    hold_ctr = 0

            # ── Phase machine (AMR control) ───────────────────────────────────
            if phase == Phase.DRIVE_TO_LOAD:
                if (np.linalg.norm(pos - np.array(waypoints[wp_idx])) < WAYPOINT_R
                        and wp_idx < len(waypoints) - 1):
                    wp_idx += 1
                arrived = amr.drive(data, waypoints[wp_idx], ranges)
                if arrived or np.linalg.norm(pos - LOAD_ZONE) < ARRIVE_R:
                    amr.stop(data)
                    print(f"\n  [t={data.time:.1f}s] AMR at load zone — "
                          f"waiting for Franka #1\n")
                    phase = Phase.WAIT_FOR_PKG

            elif phase == Phase.WAIT_FOR_PKG:
                amr.stop(data)

            elif phase == Phase.LOAD_PAYLOAD:
                amr.stop(data)
                ok, msg = payload.load(
                    pending_pkg["item_name"], model, data, amr.cargo_body)
                print(f"  Payload: {'OK' if ok else 'FAIL'}  {msg}")
                model.geom_rgba[amr.cargo_geom] = pending_pkg["rgba"]
                ps.set_loaded_pkg_pos(data, data.xpos[amr.cargo_body].copy())

                grid[:] = False; total_hits = 0; replan_count = 0
                raw = astar(grid, tuple(pos), tuple(DELIVERY_GOAL))
                waypoints = smooth_path(raw) if raw else [tuple(DELIVERY_GOAL)]
                wp_idx    = min(1, len(waypoints) - 1)
                print(f"\n  [t={data.time:.1f}s] Heading to delivery goal "
                      f"{tuple(DELIVERY_GOAL)}\n")
                phase = Phase.NAVIGATE

            elif phase == Phase.NAVIGATE:
                if (np.linalg.norm(pos - np.array(waypoints[wp_idx])) < WAYPOINT_R
                        and wp_idx < len(waypoints) - 1):
                    wp_idx += 1
                amr.drive(data, waypoints[wp_idx], ranges)
                payload.read_motor_effort(data)
                ps.set_loaded_pkg_pos(data, data.xpos[amr.cargo_body].copy())

                if np.linalg.norm(DELIVERY_GOAL - pos) < ARRIVE_R:
                    amr.stop(data)
                    print(f"\n  *** AMR ARRIVED AT DELIVERY ZONE ***  "
                          f"t={data.time:.1f}s  "
                          f"pos=({pos[0]:.2f},{pos[1]:.2f})  "
                          f"replans={replan_count}\n")
                    print(payload.status())
                    phase = Phase.STAGE3_UNLOAD
                    hold_ctr = 0

            elif phase == Phase.STAGE3_UNLOAD:
                amr.stop(data)
                if hold_ctr == 0:
                    print("\n  === STAGE 3: TRUCK LOADING  ===\n")
                    # Hide AMR cargo geom
                    model.geom_rgba[amr.cargo_geom] = [0.0, 0.0, 0.0, 0.0]
                    # Hide stage1 package mocap
                    if pending_pkg is not None and ps._active_pkg is not None:
                        pkg = ps._active_pkg["package"]
                        if pkg.loaded:
                            data.mocap_pos[ps._pkg_mocap[pkg.id]] = [0.0, 0.0, -10.0]
                    # Activate stage3: place package on staging table
                    stage3.activate(pending_pkg, data)
                hold_ctr += 1
                # stage3.step() is already called above (before phase machine)

            elif phase == Phase.COMPLETE:
                amr.stop(data)
                hold_ctr += 1
                if hold_ctr == 1:
                    print(f"\n  *** WAREHOUSE PIPELINE COMPLETE ***\n")
                    print(f"  Packages picked    : {ps._pkg_count}")
                    print(f"  Packages in truck  : {stage3.get_packages_loaded()}")
                    print(f"  Payload mass       : {payload.current_weight:.1f} kg")
                    print(f"  A* replans         : {replan_count}")
                    print(f"  Total sim time     : {data.time:.1f}s\n")
                if hold_ctr > 600:
                    break

            # ── LiDAR visualisation ───────────────────────────────────────────
            if hasattr(viewer, 'user_scn') and viewer.user_scn.maxgeom > 0:
                draw_lidar(viewer.user_scn, pos, ranges, amr.yaw(data))

            if step % 1000 == 0 and step > 0:
                print(f"  t={data.time:6.1f}s  phase={phase.name:16s}  "
                      f"AMR=({pos[0]:.2f},{pos[1]:.2f})  {ps.get_status()}")

            step += 1
            mujoco.mj_step(model, data)
            viewer.sync()

    print(f"\nSimulation ended.  replans={replan_count}  obstacle cells={total_hits}\n")


if __name__ == "__main__":
    main()
