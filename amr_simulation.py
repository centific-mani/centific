"""
AMR (Autonomous Mobile Robot) MuJoCo Simulation — DYNAMIC navigation
Robot navigates from (0,0) to (4,4) with NO prior map.

How it works:
  1. Grid starts completely empty  (robot knows nothing)
  2. Every step: LiDAR hits → obstacle cells marked on the grid
  3. Whenever the grid changes  → A* re-plans from current position to goal
  4. Robot follows the latest waypoint list
  5. New obstacles mid-path trigger an instant re-plan

Sensors:
  - 2-D LiDAR : 36-ray 360° scan, drawn as colour-coded capsules
  - Depth camera: forward-facing; stats printed to terminal

Run:  python amr_simulation.py
"""

import heapq
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
    <material name="obs_r_mat" rgba="0.75 0.28 0.18 0.90"/>
    <material name="obs_g_mat" rgba="0.22 0.62 0.35 0.90"/>
    <material name="obs_b_mat" rgba="0.20 0.40 0.90 0.90"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="8 8 0.1"
          material="floor_mat" contype="0" conaffinity="0" pos="2 2 0"/>

    <geom name="goal"  type="cylinder" size="0.18 0.01" pos="4 4 0.01"
          material="goal_mat"  contype="0" conaffinity="0"/>
    <geom name="start" type="cylinder" size="0.12 0.01" pos="0 0 0.01"
          material="trail_mat" contype="0" conaffinity="0"/>

    <!-- contype="2" → detected by mj_ray; robot has contype=0 → no collision -->
    <geom name="obs_a"    type="box"      size="0.30 0.30 0.45"
          pos="1.0 3.0 0.45" material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_b"    type="box"      size="0.25 0.40 0.50"
          pos="3.2 1.0 0.50" material="obs_g_mat" contype="2" conaffinity="0"/>
    <geom name="obs_c"    type="cylinder" size="0.20 0.40"
          pos="2.0 3.6 0.40" material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_d"    type="box"      size="0.35 0.20 0.35"
          pos="0.6 2.0 0.35" material="obs_g_mat" contype="2" conaffinity="0"/>
    <geom name="obs_test" type="box"      size="0.50 0.50 0.60"
          pos="2.0 2.0 0.60" material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_e"    type="box"      size="0.22 0.25 0.40"
          pos="3.8 2.0 0.40" material="obs_b_mat" contype="2" conaffinity="0"/>
    <geom name="obs_f"    type="cylinder" size="0.22 0.35"
          pos="1.5 1.0 0.35" material="obs_b_mat" contype="2" conaffinity="0"/>
    <geom name="obs_g"    type="box"      size="0.25 0.22 0.38"
          pos="2.8 3.5 0.38" material="obs_b_mat" contype="2" conaffinity="0"/>

    <body name="base" pos="0 0 0.12">
      <joint name="slide_x"   type="slide" axis="1 0 0" damping="5"/>
      <joint name="slide_y"   type="slide" axis="0 1 0" damping="5"/>
      <joint name="hinge_yaw" type="hinge" axis="0 0 1" damping="3"/>

      <camera name="depth_cam" pos="0.22 0 0.05" xyaxes="0 -1 0 0 0 1"/>

      <geom name="chassis"    type="box"      size="0.22 0.16 0.07"
            material="body_mat" contype="0" conaffinity="0" group="1"/>
      <geom name="sensor_top" type="cylinder" size="0.06 0.04" pos="0.10 0 0.07"
            material="body_mat" contype="0" conaffinity="0" group="1"/>

      <body name="wheel_left" pos="-0.05 0.18 -0.05">
        <joint name="wheel_left_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom name="wl" type="cylinder" size="0.07 0.04"
              material="wheel_mat" contype="0" conaffinity="0"
              euler="90 0 0" group="1"/>
      </body>

      <body name="wheel_right" pos="-0.05 -0.18 -0.05">
        <joint name="wheel_right_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom name="wr" type="cylinder" size="0.07 0.04"
              material="wheel_mat" contype="0" conaffinity="0"
              euler="90 0 0" group="1"/>
      </body>

      <geom name="caster" type="sphere" size="0.04" pos="0.18 0 -0.05"
            material="wheel_mat" contype="0" conaffinity="0" group="1"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="act_x"   joint="slide_x"   kv="120"/>
    <velocity name="act_y"   joint="slide_y"   kv="120"/>
    <velocity name="act_yaw" joint="hinge_yaw" kv="80"/>
  </actuator>
</mujoco>
"""

# ── Sensor config ──────────────────────────────────────────────────────────────
LIDAR_RAYS    = 36
LIDAR_RANGE   = 5.0
LIDAR_Z       = 0.14
_ENV_GRP      = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
_LIDAR_ANGLES = np.linspace(0, 2 * np.pi, LIDAR_RAYS, endpoint=False)
_GEOMID_BUF   = np.array([-1], dtype=np.int32)

DEPTH_H, DEPTH_W = 64, 64
DEPTH_INTERVAL   = 2000

# ── Controller config ──────────────────────────────────────────────────────────
GOAL       = np.array([4.0, 4.0])
MAX_VEL    = 0.35
MAX_YAW    = 1.5
K_P_POS    = 2.0
K_P_YAW    = 3.0
ARRIVE_R   = 0.15
WAYPOINT_R = 0.22
WHEEL_R    = 0.07
SAFETY_R   = 0.55   # LiDAR range below which repulsion kicks in

# ── Dynamic grid config ────────────────────────────────────────────────────────
GRID_RES  = 0.05    # m per cell
GRID_COLS = 100     # 0 → 5 m in X
GRID_ROWS = 100     # 0 → 5 m in Y
INFLATE_R = 0.28    # inflation radius around each LiDAR hit point


# ── Occupancy grid helpers ─────────────────────────────────────────────────────
def _cell_centre(cx, cy):
    return (cx + 0.5) * GRID_RES, (cy + 0.5) * GRID_RES

def _world_to_cell(wx, wy):
    cx = int(np.clip(wx / GRID_RES, 0, GRID_COLS - 1))
    cy = int(np.clip(wy / GRID_RES, 0, GRID_ROWS - 1))
    return cx, cy


def update_grid_from_lidar(grid, pos2d, ranges):
    """
    For every LiDAR ray that hit an obstacle, mark that hit cell and
    surrounding cells (within INFLATE_R) as occupied.
    Returns True if any new cell was marked (path re-plan needed).
    """
    changed  = False
    n_cells  = int(np.ceil(INFLATE_R / GRID_RES))

    for angle, r in zip(_LIDAR_ANGLES, ranges):
        if r >= LIDAR_RANGE - 0.05:
            continue                         # ray hit nothing — skip

        # World coordinates of the hit point
        hx = pos2d[0] + r * np.cos(angle)
        hy = pos2d[1] + r * np.sin(angle)
        hcx, hcy = _world_to_cell(hx, hy)

        # Inflate: mark all cells within INFLATE_R of the hit point
        for dy in range(-n_cells, n_cells + 1):
            for dx in range(-n_cells, n_cells + 1):
                nx, ny = hcx + dx, hcy + dy
                if not (0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS):
                    continue
                wx, wy = _cell_centre(nx, ny)
                if (wx - hx) ** 2 + (wy - hy) ** 2 <= INFLATE_R ** 2:
                    if not grid[ny, nx]:
                        grid[ny, nx] = True
                        changed = True
    return changed


# ── A* path planner ────────────────────────────────────────────────────────────
_DIRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]

def astar(grid, start_world, goal_world):
    """
    A* from start_world to goal_world on the current occupancy grid.
    The start cell is always treated as free (robot is there).
    Returns list of world-frame (x, y) waypoints, or None if no path exists.
    """
    sc = _world_to_cell(*start_world)
    gc = _world_to_cell(*goal_world)

    def h(cx, cy):
        return np.hypot(cx - gc[0], cy - gc[1])

    g_cost = {sc: 0.0}
    parent = {sc: None}
    heap   = [(h(*sc), sc)]

    while heap:
        _, curr = heapq.heappop(heap)
        if curr == gc:
            path, node = [], curr
            while node is not None:
                path.append(_cell_centre(*node))
                node = parent[node]
            path.reverse()
            return path

        cx, cy = curr
        for dx, dy, cost in _DIRS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS):
                continue
            # Always allow the start cell; skip other blocked cells
            if grid[ny, nx] and (nx, ny) != sc:
                continue
            nc = (nx, ny)
            ng = g_cost[curr] + cost
            if nc not in g_cost or ng < g_cost[nc]:
                g_cost[nc] = ng
                parent[nc] = curr
                heapq.heappush(heap, (ng + h(nx, ny), nc))

    return None


def smooth_path(path, step=6):
    if len(path) <= 2:
        return list(path)
    pts = [path[0]]
    for i in range(step, len(path) - 1, step):
        pts.append(path[i])
    pts.append(path[-1])
    return pts


# ── Navigation helpers ─────────────────────────────────────────────────────────
def get_pos(data): return data.qpos[0:2].copy()
def get_yaw(data): return float(data.qpos[2])
def angle_wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def compute_controls(data, waypoint, ranges):
    pos  = get_pos(data)
    err  = np.array(waypoint) - pos
    dist = np.linalg.norm(err)

    if dist < 1e-3:
        return np.zeros(3), 0.0

    desired_yaw = np.arctan2(err[1], err[0])
    speed       = np.clip(K_P_POS * dist, 0.0, MAX_VEL)

    # Goal-biased repulsion when LiDAR sees something close
    min_idx   = int(np.argmin(ranges))
    min_range = float(ranges[min_idx])
    if min_range < SAFETY_R:
        obs_angle = float(_LIDAR_ANGLES[min_idx])
        repulsion = np.array([-np.cos(obs_angle), -np.sin(obs_angle)])
        goal_dir  = err / (dist + 1e-9)
        w         = 1.0 - min_range / SAFETY_R
        blend     = repulsion * w + goal_dir * (1.0 - w * 0.5)
        norm      = np.linalg.norm(blend)
        if norm > 1e-6:
            blend /= norm
        desired_yaw = np.arctan2(blend[1], blend[0])
        speed       = np.clip(K_P_POS * dist, 0.0, MAX_VEL * (min_range / SAFETY_R))

    yaw_err = angle_wrap(desired_yaw - get_yaw(data))
    vx      = speed * np.cos(desired_yaw)
    vy      = speed * np.sin(desired_yaw)
    v_yaw   = np.clip(K_P_YAW * yaw_err, -MAX_YAW, MAX_YAW)

    return np.array([vx, vy, v_yaw]), np.linalg.norm([vx, vy]) / WHEEL_R


# ── 2-D LiDAR ─────────────────────────────────────────────────────────────────
def scan_lidar(model, data, pos2d):
    origin = np.array([pos2d[0], pos2d[1], LIDAR_Z])
    ranges = np.full(LIDAR_RAYS, LIDAR_RANGE)
    for i, a in enumerate(_LIDAR_ANGLES):
        vec = np.array([np.cos(a), np.sin(a), 0.0])
        d   = mujoco.mj_ray(model, data, origin, vec, _ENV_GRP, 1, -1, _GEOMID_BUF)
        if 0.0 < d < LIDAR_RANGE:
            ranges[i] = d
    return ranges


def _add_ray_capsule(scn, origin, angle, dist, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    end    = origin + np.array([np.cos(angle), np.sin(angle), 0.0]) * dist
    mid    = (origin + end) * 0.5
    z_axis = end - origin
    length = np.linalg.norm(z_axis)
    if length < 1e-6:
        return
    z_axis /= length
    ref    = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(z_axis, ref);  y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    g           = scn.geoms[scn.ngeom]
    g.type      = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.size[:]   = [0.009, length * 0.5, 0.009]
    g.pos[:]    = mid
    g.mat[:, :] = np.column_stack([x_axis, y_axis, z_axis])
    g.rgba[:]   = rgba
    scn.ngeom  += 1


def draw_lidar(scn, pos2d, ranges):
    scn.ngeom = 0
    origin    = np.array([pos2d[0], pos2d[1], LIDAR_Z])
    for angle, r in zip(_LIDAR_ANGLES, ranges):
        t    = r / LIDAR_RANGE
        rgba = np.array([1.0 - t, t * 0.85, 0.0, 0.82], dtype=np.float32)
        _add_ray_capsule(scn, origin, angle, r, rgba)


# ── Depth camera ───────────────────────────────────────────────────────────────
def print_depth_stats(renderer, data, step):
    renderer.update_scene(data, camera="depth_cam")
    depth = renderer.render()
    valid = depth[(depth > 0.05) & (depth < 1e5)]
    if valid.size:
        print(f"  [t={data.time:7.1f}s | step {step:6d}]  "
              f"DepthCam  min={valid.min():.2f}m  "
              f"mean={valid.mean():.2f}m  "
              f"max={valid.max():.2f}m  "
              f"({valid.size}/{depth.size} valid px)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    model    = mujoco.MjModel.from_xml_string(MODEL_XML)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, DEPTH_H, DEPTH_W)
    renderer.enable_depth_rendering()
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # ── Dynamic map: starts completely empty ───────────────────────────────────
    grid       = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    total_hits = 0

    # Initial A* on the empty grid → straight-line path to goal
    print("Dynamic navigation starting — grid is empty, no obstacle knowledge.")
    raw_path  = astar(grid, (0.0, 0.0), tuple(GOAL))
    waypoints = smooth_path(raw_path)
    wp_idx    = min(1, len(waypoints) - 1)
    print(f"  Initial plan: {len(waypoints)} waypoints (straight line, no obstacles known yet)")
    print()

    arrived     = False
    hold_time   = 0.0
    HOLD_SECS   = 2.0
    step        = 0
    replan_count = 0
    ranges      = np.full(LIDAR_RAYS, LIDAR_RANGE)

    print(f"  Target   : {GOAL}")
    print(f"  Max vel  : {MAX_VEL} m/s")
    print(f"  LiDAR    : {LIDAR_RAYS} rays / {360 // LIDAR_RAYS}° / {LIDAR_RANGE} m range")
    print(f"  Grid     : {GRID_COLS}×{GRID_ROWS} cells @ {GRID_RES} m  inflate={INFLATE_R} m")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0]
        viewer.cam.distance  = 9.5
        viewer.cam.elevation = -35
        viewer.cam.azimuth   = 45

        while viewer.is_running():
            pos    = get_pos(data)

            # ── 1. Sense: LiDAR scan ───────────────────────────────────────────
            ranges = scan_lidar(model, data, pos)

            # ── 2. Map: update occupancy grid from LiDAR hits ─────────────────
            grid_changed = update_grid_from_lidar(grid, pos, ranges)
            if grid_changed:
                new_hits    = int(grid.sum()) - total_hits
                total_hits  = int(grid.sum())

                # ── 3. Plan: re-run A* from current position ──────────────────
                raw_path = astar(grid, tuple(pos), tuple(GOAL))
                if raw_path is not None:
                    waypoints    = smooth_path(raw_path)
                    wp_idx       = min(1, len(waypoints) - 1)
                    replan_count += 1
                    print(f"  [replan #{replan_count:3d} | t={data.time:6.1f}s | "
                          f"step {step:5d}]  "
                          f"+{new_hits} new cells  →  "
                          f"{len(waypoints)} waypoints  "
                          f"occupied={total_hits}")
                else:
                    print(f"  [t={data.time:.1f}s] WARNING: no path found — holding current route")

            # ── 4. Act: follow current waypoint list ──────────────────────────
            if not arrived:
                waypoint = waypoints[wp_idx]
                if (np.linalg.norm(pos - np.array(waypoint)) < WAYPOINT_R
                        and wp_idx < len(waypoints) - 1):
                    wp_idx += 1
                    waypoint = waypoints[wp_idx]

                ctrl, wheel_spin = compute_controls(data, waypoint, ranges)
                data.ctrl[:]  = ctrl
                data.qvel[3]  =  wheel_spin
                data.qvel[4]  = -wheel_spin

                if np.linalg.norm(GOAL - pos) < ARRIVE_R:
                    arrived      = True
                    data.ctrl[:] = 0.0
                    print(f"\n  *** Arrived at goal!  "
                          f"pos=({pos[0]:.3f},{pos[1]:.3f})  "
                          f"t={data.time:.1f}s  "
                          f"replans={replan_count}  "
                          f"map cells={total_hits} ***")
            else:
                data.qvel[3]  = 0.0
                data.qvel[4]  = 0.0
                hold_time    += model.opt.timestep
                if hold_time > HOLD_SECS:
                    print("  Done. Closing viewer.")
                    break

            # ── Visualise ──────────────────────────────────────────────────────
            if hasattr(viewer, 'user_scn') and viewer.user_scn.maxgeom > 0:
                draw_lidar(viewer.user_scn, pos, ranges)

            if step % DEPTH_INTERVAL == 0:
                print_depth_stats(renderer, data, step)

            step += 1
            mujoco.mj_step(model, data)
            viewer.sync()

    print(f"\nSimulation finished.  Total replans: {replan_count}  "
          f"Final map: {total_hits} occupied cells.")


if __name__ == "__main__":
    main()
