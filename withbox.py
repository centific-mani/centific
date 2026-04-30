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

Run:  python amr_simulation.py
"""
import numpy as np
import mujoco
import mujoco.viewer
import heapq

# ── XML model ──────────────────────────────────────────────────────────────────
# ── Controller config ──────────────────────────────────────────────────────────
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
LIDAR_RANGE    = 5.0          # metres
LIDAR_Z        = 0.14         # scan-plane height (just above chassis top)
# geomgroup mask: include group-0 (environment) only, skip group-1 (robot)
_ENV_GRP       = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
_LIDAR_ANGLES  = np.linspace(0, 2 * np.pi, LIDAR_RAYS, endpoint=False)
_GEOMID_BUF    = np.array([-1], dtype=np.int32)

DEPTH_H, DEPTH_W   = 64, 64
DEPTH_INTERVAL     = 2000     # print depth stats every N sim steps

# ── Controller config ──────────────────────────────────────────────────────────
GOAL      = np.array([4.0,4.0])
MAX_VEL   = 0.2            # m/s
MAX_YAW   = 1.2               # rad/s
K_P_POS   = 1.0
K_P_YAW   = 2.5
ARRIVE_R  = 0.12
WHEEL_R   = 0.07
OBSTACLE_THRESHOLD = 1.0   # meters - avoid obstacles closer than this (LiDAR)
DEPTH_OBSTACLE_THRESHOLD = 0.5  # meters - forward obstacle detection (depth camera)

# ── A* Path Planning ────────────────────────────────────────────────────────────
GRID_SIZE = 5  # 5x5 grid, each cell 1m x 1m
CELL_SIZE = 1.0

# Obstacle cells (based on XML positions)
OBSTACLE_CELLS = {
    (1, 3),  # obs_a at (1,3)
    (3, 1),  # obs_b at (3,1)
    (2, 3),  # obs_c at (2,3)
    (0, 2),  # obs_d at (0,2)
    (2, 2),  # obs_test at (2,2)
}

def grid_pos(world_pos):
    """Convert world position to grid coordinates."""
    return int(world_pos[0] // CELL_SIZE), int(world_pos[1] // CELL_SIZE)

def world_pos(grid_pos):
    """Convert grid coordinates to world position (center of cell)."""
    return (grid_pos[0] + 0.5) * CELL_SIZE, (grid_pos[1] + 0.5) * CELL_SIZE

def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal):
    """A* algorithm to find path from start to goal grid positions."""
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current = heapq.heappop(frontier)[1]

        if current == goal:
            break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-directional
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                if neighbor in OBSTACLE_CELLS:
                    continue
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(goal, neighbor)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

    # Reconstruct path
    if goal not in came_from:
        return []  # No path found

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

# ── Navigation helpers ─────────────────────────────────────────────────────────
def get_pos(data):  return data.qpos[0:2].copy()
def get_yaw(data):  return float(data.qpos[2])
def angle_wrap(a):  return (a + np.pi) % (2 * np.pi) - np.pi


def compute_controls(data, ranges, depth_obstacle, min_depth, goal):
    pos  = get_pos(data)
    err  = goal - pos
    dist = np.linalg.norm(err)
    if dist < ARRIVE_R:
        return np.zeros(3), 0.0
    desired_yaw = np.arctan2(err[1], err[0])
    yaw_err     = angle_wrap(desired_yaw - get_yaw(data))
    speed = np.clip(K_P_POS * dist, 0.0, MAX_VEL)
    vx    = speed * np.cos(desired_yaw)
    vy    = speed * np.sin(desired_yaw)
    v_yaw = np.clip(K_P_YAW * yaw_err, -MAX_YAW, MAX_YAW)

    # Depth camera: detect forward obstacle
    if depth_obstacle:
        print(f"Depth Camera: Obstacle ahead at {min_depth:.2f}m, stopping forward motion...")
        speed = 0.0  # stop forward motion
        vx = 0.0
        vy = 0.0
        v_yaw = 1.0  # turn to find way around
        return np.array([vx, vy, v_yaw]), 0.0
    
    # Obstacle avoidance using LiDAR (360° awareness)
    min_range = np.min(ranges)
    if min_range < OBSTACLE_THRESHOLD:
        print(f"LiDAR: Obstacle detected at {min_range:.2f}m (angle: {_LIDAR_ANGLES[np.argmin(ranges)]:.2f} rad), avoiding...")
        # Strong avoidance: turn 90 degrees away from obstacle
        min_idx = np.argmin(ranges)
        obstacle_angle = _LIDAR_ANGLES[min_idx]
        desired_yaw = obstacle_angle + np.pi / 2  # perpendicular turn
        yaw_err = angle_wrap(desired_yaw - get_yaw(data))
        speed = 0.05  # reduced speed for careful avoidance
        vx = speed * np.cos(desired_yaw)
        vy = speed * np.sin(desired_yaw)
        v_yaw = np.clip(2.0 * yaw_err, -MAX_YAW, MAX_YAW)  # stronger turning gain

    return np.array([vx, vy, v_yaw]), np.linalg.norm([vx, vy]) / WHEEL_R


# ── 2-D LiDAR ─────────────────────────────────────────────────────────────────
def scan_lidar(model, data, pos2d):
    """Cast 36 horizontal rays; return per-ray distances (capped at LIDAR_RANGE)."""
    origin = np.array([pos2d[0], pos2d[1], LIDAR_Z])
    ranges = np.full(LIDAR_RAYS, LIDAR_RANGE)
    for i, a in enumerate(_LIDAR_ANGLES):
        vec = np.array([np.cos(a), np.sin(a), 0.0])
        d   = mujoco.mj_ray(model, data, origin, vec, _ENV_GRP, 1, -1, _GEOMID_BUF)
        if 0.0 < d < LIDAR_RANGE:
            ranges[i] = d
    return ranges


def _add_ray_capsule(scn, origin, angle, dist, rgba):
    """Append one ray as a thin colour-coded capsule to viewer.user_scn."""
    if scn.ngeom >= scn.maxgeom:
        return
    end    = origin + np.array([np.cos(angle), np.sin(angle), 0.0]) * dist
    mid    = (origin + end) * 0.5
    z_axis = end - origin
    length = np.linalg.norm(z_axis)
    if length < 1e-6:
        return
    z_axis /= length
    # Build orthonormal frame with z_axis as the capsule long-axis
    ref    = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(z_axis, ref);  y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    g           = scn.geoms[scn.ngeom]
    g.type      = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.size[:]   = [0.009, length * 0.5, 0.009]
    g.pos[:]    = mid
    # mat is (3,3) row-major: columns are the frame axes
    g.mat[:, :] = np.column_stack([x_axis, y_axis, z_axis])
    g.rgba[:]   = rgba
    scn.ngeom  += 1


def draw_lidar(scn, pos2d, ranges):
    """Redraw all LiDAR rays: green=far, red=close."""
    scn.ngeom = 0
    origin    = np.array([pos2d[0], pos2d[1], LIDAR_Z])
    for angle, r in zip(_LIDAR_ANGLES, ranges):
        t    = r / LIDAR_RANGE            # 0 = close (red), 1 = far (green)
        rgba = np.array([1.0 - t, t * 0.85, 0.0, 0.82], dtype=np.float32)
        _add_ray_capsule(scn, origin, angle, r, rgba)


# ── Depth camera ───────────────────────────────────────────────────────────────
def get_depth_forward_obstacle(renderer, data):
    """Check if there's an obstacle directly ahead using depth camera.
    Returns: (has_obstacle, min_depth_ahead)"""
    renderer.update_scene(data, camera="depth_cam")
    depth = renderer.render()  # float32 (H, W) in metres
    # Check central region (forward-facing): middle 50% of image
    h_start, h_end = DEPTH_H // 4, 3 * DEPTH_H // 4
    w_start, w_end = DEPTH_W // 4, 3 * DEPTH_W // 4
    forward_region = depth[h_start:h_end, w_start:w_end]
    valid = forward_region[(forward_region > 0.05) & (forward_region < 1e5)]
    if valid.size > 0:
        min_depth = valid.min()
        has_obstacle = min_depth < DEPTH_OBSTACLE_THRESHOLD
        return has_obstacle, min_depth
    return False, DEPTH_OBSTACLE_THRESHOLD


def print_depth_stats(renderer, data, step):
    renderer.update_scene(data, camera="depth_cam")
    depth = renderer.render()                    # float32 (H, W) in metres
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
    mujoco.mj_forward(model, data)   # populate geom_xpos so mj_ray works from frame 0

    arrived   = False
    hold_time = 0.0
    HOLD_SECS = 2.0
    step      = 0

    # Compute A* path
    start_grid = (0, 0)
    goal_grid = (4, 4)
    path = a_star(start_grid, goal_grid)
    if not path:
        print("No path found to goal!")
        return
    waypoints = [world_pos(p) for p in path]
    print(f"Planned path with {len(waypoints)} waypoints: {[f'({x:.1f},{y:.1f})' for x,y in waypoints]}")
    current_waypoint = 0

    print("AMR simulation starting ...")
    print(f"  Target   : {GOAL}")
    print(f"  Max vel  : {MAX_VEL} m/s")
    print(f"  LiDAR    : {LIDAR_RAYS} rays / {360 // LIDAR_RAYS} deg resolution / {LIDAR_RANGE} m range")
    print(f"  DepthCam : {DEPTH_W}x{DEPTH_H} px  (stats every {DEPTH_INTERVAL} steps)")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0]
        viewer.cam.distance   = 9.5
        viewer.cam.elevation  = -35
        viewer.cam.azimuth    = 45

        while viewer.is_running():
            pos = get_pos(data)

            # ── navigation ──────────────────────────────────────────────────
            if not arrived:
                if current_waypoint < len(waypoints):
                    current_goal = np.array(waypoints[current_waypoint])
                    # Get sensor data from both LiDAR and Depth Camera
                    ranges = scan_lidar(model, data, pos)
                    depth_obstacle, min_depth = get_depth_forward_obstacle(renderer, data)
                    
                    ctrl, wheel_spin = compute_controls(data, ranges, depth_obstacle, min_depth, current_goal)
                    data.ctrl[:]  = ctrl
                    data.qvel[3]  =  wheel_spin
                    data.qvel[4]  = -wheel_spin
                    if np.linalg.norm(current_goal - pos) < ARRIVE_R:
                        current_waypoint += 1
                        print(f"Reached waypoint {current_waypoint}, moving to next...")
                else:
                    arrived = True
                    data.ctrl[:] = 0.0
                    print(f"\nArrived at final goal! pos={pos} t={data.time:.1f}s")
            else:
                data.qvel[3] = 0.0
                data.qvel[4] = 0.0
                hold_time   += model.opt.timestep
                if hold_time > HOLD_SECS:
                    print("  Done. Closing viewer.")
                    break

            # ── 2-D LiDAR scan + visualise ──────────────────────────────────
            if hasattr(viewer, 'user_scn') and viewer.user_scn.maxgeom > 0:
                draw_lidar(viewer.user_scn, pos, ranges)

            # ── depth camera (periodic terminal stats) ───────────────────────
            if step % DEPTH_INTERVAL == 0:
                print_depth_stats(renderer, data, step)

            step += 1
            mujoco.mj_step(model, data)
            viewer.sync()

    print("Simulation finished.")


if __name__ == "__main__":
    main()
