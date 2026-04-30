"""
AMR (Autonomous Mobile Robot) MuJoCo Simulation
Differential-drive robot navigating from (0,0) to (4,4).

Architecture:
  - Three planar DOFs (slide-X, slide-Y, hinge-Yaw) for stable body control
  - Moderate-gain velocity servos drive the body directly in world frame
  - Two wheel hinges animate the spinning wheels visually
  - All collisions disabled; floor is purely visual
  - 2-D LiDAR : 36-ray 360 deg scan drawn in the viewer as colour-coded capsules
  - Depth camera: forward-facing; depth-image stats printed to terminal
  - A* path planner on a 2D occupancy grid with inflated obstacle margins
  - LiDAR goal-biased repulsion as a dynamic safety layer

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
    <!-- Floor - visual only, no collision -->
    <geom name="floor" type="plane" size="8 8 0.1"
          material="floor_mat" contype="0" conaffinity="0" pos="2 2 0"/>

    <!-- Goal / start markers -->
    <geom name="goal"  type="cylinder" size="0.18 0.01" pos="4 4 0.01"
          material="goal_mat"  contype="0" conaffinity="0"/>
    <geom name="start" type="cylinder" size="0.12 0.01" pos="0 0 0.01"
          material="trail_mat" contype="0" conaffinity="0"/>

    <!-- Obstacles: contype="2" makes mj_ray detect them, but the robot
         has contype=0/conaffinity=0 so (2&0)|(0&2)=0 -- no physical contact. -->
    <geom name="obs_a" type="box"      size="0.30 0.30 0.45"
          pos="1.0 3.0 0.45" material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_b" type="box"      size="0.25 0.40 0.50"
          pos="3.2 1.0 0.50" material="obs_g_mat" contype="2" conaffinity="0"/>
    <geom name="obs_c" type="cylinder" size="0.20 0.40"
          pos="2.0 3.6 0.40" material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_d" type="box"      size="0.35 0.20 0.35"
          pos="0.6 2.0 0.35" material="obs_g_mat" contype="2" conaffinity="0"/>
    <!-- Test obstacle in direct path -->
    <geom name="obs_test" type="box" size="0.50 0.50 0.60"
          pos="2.0 2.0 0.60" material="obs_r_mat" contype="2" conaffinity="0"/>

    <!-- Extra obstacles to stress-test avoidance -->
    <!-- obs_e: blocks right-side corridor at mid-height -->
    <geom name="obs_e" type="box"      size="0.22 0.25 0.40"
          pos="3.8 2.0 0.40" material="obs_b_mat" contype="2" conaffinity="0"/>
    <!-- obs_f: cylinder in lower-left quadrant -->
    <geom name="obs_f" type="cylinder" size="0.22 0.35"
          pos="1.5 1.0 0.35" material="obs_b_mat" contype="2" conaffinity="0"/>
    <!-- obs_g: box in upper-right area near goal approach -->
    <geom name="obs_g" type="box"      size="0.25 0.22 0.38"
          pos="2.8 3.5 0.38" material="obs_b_mat" contype="2" conaffinity="0"/>

    <!-- AMR body - three free planar DOFs -->
    <body name="base" pos="0 0 0.12">
      <joint name="slide_x"   type="slide" axis="1 0 0" damping="5"/>
      <joint name="slide_y"   type="slide" axis="0 1 0" damping="5"/>
      <joint name="hinge_yaw" type="hinge" axis="0 0 1" damping="3"/>

      <!-- Depth camera forward-facing (+X body direction).
           xyaxes: cam-X=(0,-1,0)  cam-Y=(0,0,1)  => cam-Z=(-1,0,0)
           Camera looks along -cam-Z = +X_body. -->
      <camera name="depth_cam" pos="0.22 0 0.05" xyaxes="0 -1 0 0 0 1"/>

      <!-- Robot geoms in group 1 so LiDAR ray-casts skip them -->
      <geom name="chassis"    type="box"      size="0.22 0.16 0.07"
            material="body_mat" contype="0" conaffinity="0" group="1"/>
      <geom name="sensor_top" type="cylinder" size="0.06 0.04" pos="0.10 0 0.07"
            material="body_mat" contype="0" conaffinity="0" group="1"/>

      <!-- Left wheel -->
      <body name="wheel_left" pos="-0.05 0.18 -0.05">
        <joint name="wheel_left_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom name="wl" type="cylinder" size="0.07 0.04"
              material="wheel_mat" contype="0" conaffinity="0"
              euler="90 0 0" group="1"/>
      </body>

      <!-- Right wheel -->
      <body name="wheel_right" pos="-0.05 -0.18 -0.05">
        <joint name="wheel_right_spin" type="hinge" axis="0 1 0" damping="0.3"/>
        <geom name="wr" type="cylinder" size="0.07 0.04"
              material="wheel_mat" contype="0" conaffinity="0"
              euler="90 0 0" group="1"/>
      </body>

      <!-- Front caster (visual) -->
      <geom name="caster" type="sphere" size="0.04" pos="0.18 0 -0.05"
            material="wheel_mat" contype="0" conaffinity="0" group="1"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Body velocity servos (world frame) only - wheels animated via qvel -->
    <velocity name="act_x"   joint="slide_x"   kv="120"/>
    <velocity name="act_y"   joint="slide_y"   kv="120"/>
    <velocity name="act_yaw" joint="hinge_yaw" kv="80"/>
  </actuator>
</mujoco>
"""

# ── Sensor config ──────────────────────────────────────────────────────────────
LIDAR_RAYS     = 36
LIDAR_RANGE    = 5.0
LIDAR_Z        = 0.14
_ENV_GRP       = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
_LIDAR_ANGLES  = np.linspace(0, 2 * np.pi, LIDAR_RAYS, endpoint=False)
_GEOMID_BUF    = np.array([-1], dtype=np.int32)

DEPTH_H, DEPTH_W = 64, 64
DEPTH_INTERVAL   = 2000

# ── Controller config ──────────────────────────────────────────────────────────
GOAL       = np.array([4.0, 4.0])
MAX_VEL    = 0.35          # m/s  (increased from 0.1 for practical travel time)
MAX_YAW    = 1.5           # rad/s
K_P_POS    = 2.0
K_P_YAW    = 3.0
ARRIVE_R   = 0.15          # final goal arrival radius (m)
WAYPOINT_R = 0.22          # advance to next waypoint when within this radius (m)
WHEEL_R    = 0.07
SAFETY_R   = 0.55          # LiDAR range below which goal-biased repulsion kicks in

# ── A* planner config ──────────────────────────────────────────────────────────
GRID_RES  = 0.05    # metres per grid cell  → 100×100 for a 5×5 m arena
GRID_COLS = 100
GRID_ROWS = 100
INFLATE_R = 0.25    # obstacle inflation radius (≥ robot half-width 0.22 m)

# Static obstacle geometry (must mirror MODEL_XML exactly).
# Boxes: (cx, cy, half_x, half_y)
# Cylinders: (cx, cy, radius)
_BOXES = [
    (1.0, 3.0, 0.30, 0.30),   # obs_a
    (3.2, 1.0, 0.25, 0.40),   # obs_b
    (0.6, 2.0, 0.35, 0.20),   # obs_d
    (2.0, 2.0, 0.50, 0.50),   # obs_test
    (3.8, 2.0, 0.22, 0.25),   # obs_e  – right corridor mid-height
    (2.8, 3.5, 0.25, 0.22),   # obs_g  – upper-right near goal approach
]
_CYLINDERS = [
    (2.0, 3.6, 0.20),         # obs_c
    (1.5, 1.0, 0.22),         # obs_f  – lower-left quadrant
]


# ── Occupancy grid ─────────────────────────────────────────────────────────────
def _cell_centre(cx, cy):
    return (cx + 0.5) * GRID_RES, (cy + 0.5) * GRID_RES

def _world_to_cell(wx, wy):
    return int(wx / GRID_RES), int(wy / GRID_RES)

def build_occupancy_grid():
    """Return bool array (GRID_ROWS × GRID_COLS); True = occupied / inflated."""
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for gy in range(GRID_ROWS):
        for gx in range(GRID_COLS):
            wx, wy = _cell_centre(gx, gy)
            for cx, cy, hx, hy in _BOXES:
                if abs(wx - cx) < hx + INFLATE_R and abs(wy - cy) < hy + INFLATE_R:
                    grid[gy, gx] = True
                    break
            if not grid[gy, gx]:
                for cx, cy, r in _CYLINDERS:
                    if (wx - cx) ** 2 + (wy - cy) ** 2 < (r + INFLATE_R) ** 2:
                        grid[gy, gx] = True
                        break
    return grid


# ── A* path planner ────────────────────────────────────────────────────────────
_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]

def astar(grid, start_world, goal_world):
    """
    A* on the occupancy grid.
    Returns list of world-frame (x, y) waypoints, or None if no path found.
    """
    sc = _world_to_cell(*start_world)
    gc = _world_to_cell(*goal_world)

    def h(cx, cy):
        return np.hypot(cx - gc[0], cy - gc[1])  # Euclidean (admissible)

    g_cost = {sc: 0.0}
    parent = {sc: None}
    heap   = [(h(*sc), sc)]

    while heap:
        _, curr = heapq.heappop(heap)
        if curr == gc:
            path = []
            node = curr
            while node is not None:
                path.append(_cell_centre(*node))
                node = parent[node]
            path.reverse()
            return path

        cx, cy = curr
        for dx, dy, cost in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS):
                continue
            if grid[ny, nx]:
                continue
            nc  = (nx, ny)
            ng  = g_cost[curr] + cost
            if nc not in g_cost or ng < g_cost[nc]:
                g_cost[nc] = ng
                parent[nc] = curr
                heapq.heappush(heap, (ng + h(nx, ny), nc))

    return None


def smooth_path(path, step=6):
    """Sub-sample path to reduce waypoint count; always keeps start and goal."""
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
    """
    Proportional controller toward *waypoint* with goal-biased LiDAR repulsion.
    Returns (ctrl[3], wheel_spin).
    """
    pos  = get_pos(data)
    err  = np.array(waypoint) - pos
    dist = np.linalg.norm(err)

    if dist < 1e-3:
        return np.zeros(3), 0.0

    desired_yaw = np.arctan2(err[1], err[0])
    speed       = np.clip(K_P_POS * dist, 0.0, MAX_VEL)

    # Goal-biased repulsion: blend repulsion away from closest obstacle with
    # attraction toward the waypoint, weighted by proximity.
    min_idx   = int(np.argmin(ranges))
    min_range = float(ranges[min_idx])
    if min_range < SAFETY_R:
        obs_angle = float(_LIDAR_ANGLES[min_idx])
        repulsion = np.array([-np.cos(obs_angle), -np.sin(obs_angle)])
        goal_dir  = err / (dist + 1e-9)
        # w=1 when touching threshold, 0 at threshold boundary
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
    g          = scn.geoms[scn.ngeom]
    g.type     = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.size[:]  = [0.009, length * 0.5, 0.009]
    g.pos[:]   = mid
    g.mat[:, :] = np.column_stack([x_axis, y_axis, z_axis])
    g.rgba[:]  = rgba
    scn.ngeom += 1


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

    # ── Path planning ──────────────────────────────────────────────────────────
    print("Building occupancy grid ...")
    grid     = build_occupancy_grid()
    occupied = int(grid.sum())
    print(f"  Grid {GRID_COLS}×{GRID_ROWS}  resolution {GRID_RES} m/cell  "
          f"({occupied} cells occupied, inflate={INFLATE_R} m)")

    print("Running A* ...")
    raw_path = astar(grid, (0.0, 0.0), tuple(GOAL))
    if raw_path is None:
        print("ERROR: A* found no collision-free path.  Check obstacle layout.")
        return

    waypoints = smooth_path(raw_path, step=6)
    print(f"  Raw path: {len(raw_path)} cells  →  {len(waypoints)} waypoints after smoothing")
    for i, wp in enumerate(waypoints):
        print(f"    wp[{i:2d}] = ({wp[0]:.2f}, {wp[1]:.2f})")

    # Start at wp index 1 (wp[0] ≈ robot's current position)
    wp_idx = min(1, len(waypoints) - 1)

    # ── Simulation state ───────────────────────────────────────────────────────
    arrived   = False
    hold_time = 0.0
    HOLD_SECS = 2.0
    step      = 0
    ranges    = np.full(LIDAR_RAYS, LIDAR_RANGE)   # initialised to avoid NameError

    print()
    print("AMR simulation starting ...")
    print(f"  Target   : {GOAL}")
    print(f"  Max vel  : {MAX_VEL} m/s")
    print(f"  LiDAR    : {LIDAR_RAYS} rays / {360 // LIDAR_RAYS}° resolution / {LIDAR_RANGE} m range")
    print(f"  Safety R : {SAFETY_R} m  (LiDAR repulsion threshold)")
    print(f"  DepthCam : {DEPTH_W}×{DEPTH_H} px  (stats every {DEPTH_INTERVAL} steps)")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0]
        viewer.cam.distance  = 9.5
        viewer.cam.elevation = -35
        viewer.cam.azimuth   = 45

        while viewer.is_running():
            pos    = get_pos(data)
            ranges = scan_lidar(model, data, pos)

            if not arrived:
                # Advance waypoint index when close enough
                waypoint = waypoints[wp_idx]
                if (np.linalg.norm(pos - np.array(waypoint)) < WAYPOINT_R
                        and wp_idx < len(waypoints) - 1):
                    wp_idx  += 1
                    waypoint = waypoints[wp_idx]
                    print(f"  → waypoint {wp_idx}/{len(waypoints)-1}  "
                          f"({waypoint[0]:.2f}, {waypoint[1]:.2f})  "
                          f"t={data.time:.1f}s")

                ctrl, wheel_spin = compute_controls(data, waypoint, ranges)
                data.ctrl[:]  = ctrl
                data.qvel[3]  =  wheel_spin
                data.qvel[4]  = -wheel_spin

                if np.linalg.norm(GOAL - pos) < ARRIVE_R:
                    arrived      = True
                    data.ctrl[:] = 0.0
                    print(f"\n  Arrived at goal!  pos=({pos[0]:.3f}, {pos[1]:.3f})  t={data.time:.1f}s")
            else:
                data.qvel[3]  = 0.0
                data.qvel[4]  = 0.0
                hold_time    += model.opt.timestep
                if hold_time > HOLD_SECS:
                    print("  Done. Closing viewer.")
                    break

            if hasattr(viewer, 'user_scn') and viewer.user_scn.maxgeom > 0:
                draw_lidar(viewer.user_scn, pos, ranges)

            if step % DEPTH_INTERVAL == 0:
                print_depth_stats(renderer, data, step)

            step += 1
            mujoco.mj_step(model, data)
            viewer.sync()

    print("Simulation finished.")


if __name__ == "__main__":
    main()
