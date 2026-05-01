"""
Stage 3 — Truck Loading Station
Franka Panda #2 (s3_ prefix) + LIFO Priority Planner + BLB 3D Bin Packer + RRT Motion Plan

Receives package from AMR at staging table, loads it into truck using:
  - LIFO stack ordering (last delivered = first unloaded)
  - 3D bin packing (Bottom-Left-Back) for truck placement
  - RRT joint-space motion planning

Run standalone:  python stage3_truck_loading.py
Import:          from stage3_truck_loading import TruckLoadingStation
"""

import copy
import pathlib
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import mujoco.viewer
from enum import Enum, auto
from typing import List, Optional, Tuple

PANDA_DIR = pathlib.Path(__file__).parent / "franka_emika_panda"
PANDA_XML  = PANDA_DIR / "panda.xml"

PREFIX = "s3_"

# ── Stage 3 layout ─────────────────────────────────────────────────────────────
ARM3_BASE   = (7.5, 7.0, 0.0)   # Franka #2 base position
STAGING_POS = (7.5, 6.3, 0.0)   # staging table world position
STAGING_H   = 0.65               # staging table surface height (m)
TRUCK_POS   = (8.3, 7.0, 0.0)   # truck centre position
TRUCK_BED_H = 0.30               # truck bed floor height (m)

PKG_HALF = np.array([0.12, 0.10, 0.08])   # package half-dims (m)

# Truck bed interior for BLB packer
TRUCK_BED_ORIGIN = np.array([TRUCK_POS[0] - 1.0,
                               TRUCK_POS[1] - 0.65,
                               TRUCK_BED_H + 0.025])   # (7.3, 6.35, 0.325)
TRUCK_BED_SIZE   = np.array([2.0, 1.3, 0.7])

# ── Arm joint keyframes (7 joints, radians) ────────────────────────────────────
# joint1 ≈ -π/2 → arm faces -Y (staging table)
# joint1 ≈  0   → arm faces +X (truck)
Q3_HOME      = np.array([ 0.000, -0.785,  0.000, -2.356,  0.000,  1.571,  0.785])
Q3_PRE_PICK  = np.array([-1.500,  0.400,  0.100, -1.600,  0.000,  2.000,  0.500])
Q3_PICK      = np.array([-1.500,  0.620,  0.100, -1.300,  0.000,  1.900,  0.500])
Q3_LIFT      = np.array([-1.500,  0.050,  0.100, -2.300,  0.000,  2.300,  0.500])
Q3_TRANSPORT = np.array([ 0.000,  0.200,  0.000, -2.100,  0.000,  2.300,  0.700])
Q3_DEPOSIT   = np.array([ 0.000,  0.500,  0.000, -1.800,  0.000,  2.100,  0.700])

GRIPPER_OPEN   = 255.0
GRIPPER_CLOSED =   5.0

STATE_HOLD = {
    "PRE_PICK":  520,
    "PICK":      340,
    "LIFT":      420,
    "TRANSPORT": 560,
    "DEPOSIT":   300,
    "RETRACT":   520,
}


class S3ArmState(Enum):
    IDLE      = auto()
    PRE_PICK  = auto()
    PICK      = auto()
    LIFT      = auto()
    TRANSPORT = auto()
    DEPOSIT   = auto()
    RETRACT   = auto()
    DONE      = auto()


# ── RRT joint-space motion planner ────────────────────────────────────────────
class RRTPlanner:
    """
    Rapidly-exploring Random Tree in joint space.
    Generates smooth waypoint sequences between configurations.
    Falls back to linear interpolation if max_iter exceeded.
    """

    JOINT_LIMITS = [
        (-2.8973,  2.8973),
        (-1.7628,  1.7628),
        (-2.8973,  2.8973),
        (-3.0718, -0.0698),
        (-2.8973,  2.8973),
        (-0.0175,  3.7525),
        (-2.8973,  2.8973),
    ]

    def __init__(self, step_size: float = 0.20,
                 max_iter: int = 400,
                 goal_bias: float = 0.15):
        self.step_size = step_size
        self.max_iter  = max_iter
        self.goal_bias = goal_bias
        self._rng      = np.random.default_rng(0)

    def plan(self, q_start: np.ndarray,
             q_goal: np.ndarray) -> List[np.ndarray]:
        """Return list of joint configs from q_start to q_goal."""
        q_start = np.array(q_start, dtype=float)
        q_goal  = np.array(q_goal,  dtype=float)

        nodes   = [q_start]
        parents = [-1]

        for _ in range(self.max_iter):
            if self._rng.random() < self.goal_bias:
                q_rand = q_goal.copy()
            else:
                q_rand = np.array([
                    self._rng.uniform(lo, hi)
                    for lo, hi in self.JOINT_LIMITS
                ])

            dists    = [np.linalg.norm(q_rand - n) for n in nodes]
            near_idx = int(np.argmin(dists))
            q_near   = nodes[near_idx]

            diff = q_rand - q_near
            d    = float(np.linalg.norm(diff))
            if d < 1e-6:
                continue
            q_new = q_near + self.step_size * diff / d
            nodes.append(q_new)
            parents.append(near_idx)

            if np.linalg.norm(q_new - q_goal) < self.step_size * 1.5:
                path, idx = [], len(nodes) - 1
                while idx != -1:
                    path.append(nodes[idx])
                    idx = parents[idx]
                path.reverse()
                path.append(q_goal)
                return path

        # Fallback: dense linear interpolation
        n = max(3, int(np.linalg.norm(q_goal - q_start) / self.step_size) + 1)
        return [q_start + t * (q_goal - q_start)
                for t in np.linspace(0, 1, n)]


# ── LIFO priority planner ──────────────────────────────────────────────────────
class LIFOPlanner:
    """Stack-based package ordering: last delivered = first unloaded."""

    def __init__(self):
        self._stack: List[dict] = []

    def push(self, package: dict):
        self._stack.append(package)

    def pop(self) -> Optional[dict]:
        return self._stack.pop() if self._stack else None

    def peek(self) -> Optional[dict]:
        return self._stack[-1] if self._stack else None

    def is_empty(self) -> bool:
        return len(self._stack) == 0

    def __len__(self) -> int:
        return len(self._stack)


# ── 3D bin packer (BLB) ────────────────────────────────────────────────────────
class BinPacker3D:
    """
    Bottom-Left-Back (BLB) 3D bin packing for the truck interior.
    Items are placed at the lowest, leftmost, rearmost free position.
    """

    def __init__(self,
                 bed_origin: np.ndarray = TRUCK_BED_ORIGIN,
                 bed_size:   np.ndarray = TRUCK_BED_SIZE):
        self.bed_origin = np.array(bed_origin, dtype=float)
        self.bed_size   = np.array(bed_size,   dtype=float)
        self._placed: List[Tuple[np.ndarray, np.ndarray]] = []

    def pack(self, item_size: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns world-space centre position for the item, None if truck full.
        item_size: full dimensions (not half-sizes).
        """
        size = np.array(item_size, dtype=float)

        candidates = [np.zeros(3)]
        for p_pos, p_sz in self._placed:
            candidates += [
                np.array([p_pos[0] + p_sz[0], p_pos[1],         p_pos[2]]),
                np.array([p_pos[0],            p_pos[1] + p_sz[1], p_pos[2]]),
                np.array([p_pos[0],            p_pos[1],         p_pos[2] + p_sz[2]]),
            ]

        candidates.sort(key=lambda c: (round(c[2], 3), round(c[1], 3), round(c[0], 3)))

        for pos in candidates:
            if np.all(pos >= 0) and np.all(pos + size <= self.bed_size):
                if not self._overlaps(pos, size):
                    self._placed.append((pos.copy(), size.copy()))
                    return self.bed_origin + pos + size * 0.5
        return None

    def _overlaps(self, pos: np.ndarray, size: np.ndarray) -> bool:
        for p_pos, p_sz in self._placed:
            if np.all(pos < p_pos + p_sz) and np.all(pos + size > p_pos):
                return True
        return False

    def reset(self):
        self._placed.clear()


# ── XML helpers ────────────────────────────────────────────────────────────────

def _elem_inner_xml(elem) -> str:
    """Return XML string of all children of elem (not elem itself)."""
    parts = []
    if elem.text and elem.text.strip():
        parts.append(elem.text.strip())
    for child in elem:
        parts.append(ET.tostring(child, encoding='unicode'))
    return "\n    ".join(parts)


def _prefix_panda_xml(prefix: str) -> dict:
    """
    Parse panda.xml and prefix all unique MuJoCo names with `prefix`.
    Shared names (material, mesh, default class) are NOT touched.
    Returns dict: {worldbody, actuator, tendon, equality, contact} inner XML strings.
    """
    tree = ET.parse(str(PANDA_XML))
    root = copy.deepcopy(tree.getroot())

    # Collect joint name map before any renaming
    jmap = {j.get("name"): prefix + j.get("name")
            for j in root.iter("joint") if j.get("name")}

    # Prefix body / joint / geom / site / camera names
    for tag in ("body", "joint", "geom", "site", "camera"):
        for elem in root.iter(tag):
            n = elem.get("name")
            if n:
                elem.set("name", prefix + n)

    # Prefix actuator names + joint refs
    act = root.find("actuator")
    if act is not None:
        for a in act:
            n = a.get("name")
            if n:
                a.set("name", prefix + n)
            jr = a.get("joint")
            if jr:
                a.set("joint", jmap.get(jr, prefix + jr))

    # Prefix tendon names + joint refs inside
    tendon = root.find("tendon")
    if tendon is not None:
        for t in tendon:
            n = t.get("name")
            if n:
                t.set("name", prefix + n)
            for jref in t.findall(".//joint"):
                jr = jref.get("joint")
                if jr:
                    jref.set("joint", jmap.get(jr, prefix + jr))

    # Prefix equality names + joint/body refs
    eq = root.find("equality")
    if eq is not None:
        for e in eq:
            n = e.get("name")
            if n:
                e.set("name", prefix + n)
            for attr in ("joint1", "joint2"):
                v = e.get(attr)
                if v:
                    e.set(attr, jmap.get(v, prefix + v))
            for attr in ("body1", "body2"):
                v = e.get(attr)
                if v:
                    e.set(attr, prefix + v)

    # Prefix contact body refs
    contact = root.find("contact")
    if contact is not None:
        for c in contact:
            for attr in ("body1", "body2"):
                v = c.get(attr)
                if v:
                    c.set(attr, prefix + v)

    # Remove lights from worldbody before extracting
    wb = root.find("worldbody")
    if wb is not None:
        for light in wb.findall("light"):
            wb.remove(light)

    return {
        "worldbody": _elem_inner_xml(wb)      if wb      is not None else "",
        "actuator":  _elem_inner_xml(act)     if act     is not None else "",
        "tendon":    _elem_inner_xml(tendon)  if tendon  is not None else "",
        "equality":  _elem_inner_xml(eq)      if eq      is not None else "",
        "contact":   _elem_inner_xml(contact) if contact is not None else "",
    }


# ── TruckLoadingStation ────────────────────────────────────────────────────────
class TruckLoadingStation:
    """
    Stage 3 controller: Franka Panda #2 unloads packages from staging table
    into a truck bed using LIFO ordering, BLB bin packing, and RRT motion plans.

    Integration usage
    -----------------
        ts  = TruckLoadingStation()
        snip = ts.get_xml_snippets()         # XML fragments for combined scene

        # In integrated main(), after MjModel is created:
        ts.setup(model, data)
        # When AMR arrives:
        ts.activate(pending_pkg, data)       # enqueue package; show on staging table
        # Each step during STAGE3_UNLOAD:
        done = ts.step(model, data, step)    # drives arm; True when finished
    """

    def __init__(self):
        self._lifo        = LIFOPlanner()
        self._packer      = BinPacker3D()
        self._rrt         = RRTPlanner()

        self._state       = S3ArmState.IDLE
        self._state_step  = 0
        self._target_q    = Q3_HOME.copy()
        self._gripper     = GRIPPER_OPEN
        self._active_pkg: Optional[dict] = None
        self._pkg_in_hand = False
        self._pkg_count   = 0

        # Resolved in setup()
        self._ctrl_idx: List[int] = []
        self._hand_id    = -1
        self._staged_mid = -1   # mocap id for s3_pkg_staged
        self._truck_mid  = -1   # mocap id for s3_pkg_truck
        self._staged_gid = -1   # geom id for staged pkg
        self._truck_gid  = -1   # geom id for truck pkg

    # ── XML snippets (for integration) ────────────────────────────────────────

    def get_xml_snippets(self) -> dict:
        """
        Return XML fragments to merge into the combined warehouse scene.

        Keys: worldbody, actuator, tendon, equality, contact
        Values: inner XML strings (without enclosing tags).
        """
        prefixed = _prefix_panda_xml(PREFIX)

        bx, by, bz = ARM3_BASE
        sx, sy     = STAGING_POS[0], STAGING_POS[1]
        tx, ty     = TRUCK_POS[0],   TRUCK_POS[1]
        sh         = STAGING_H
        tbh        = TRUCK_BED_H
        p          = PKG_HALF

        worldbody = f"""
    <!-- ── Stage 3: Staging table ──────────────────────────────────── -->
    <body name="staging_table" pos="{sx:.3f} {sy:.3f} 0.000">
      <geom name="staging_surface" type="box" size="0.200 0.150 0.025"
            pos="0 0 {sh:.3f}" rgba="0.70 0.55 0.30 1"
            contype="0" conaffinity="0"/>
      <geom name="staging_leg_0" type="box" size="0.018 0.018 {sh/2:.3f}"
            pos="-0.17  0.12 {sh/2:.3f}" rgba="0.45 0.35 0.20 1"
            contype="0" conaffinity="0"/>
      <geom name="staging_leg_1" type="box" size="0.018 0.018 {sh/2:.3f}"
            pos="-0.17 -0.12 {sh/2:.3f}" rgba="0.45 0.35 0.20 1"
            contype="0" conaffinity="0"/>
      <geom name="staging_leg_2" type="box" size="0.018 0.018 {sh/2:.3f}"
            pos=" 0.17  0.12 {sh/2:.3f}" rgba="0.45 0.35 0.20 1"
            contype="0" conaffinity="0"/>
      <geom name="staging_leg_3" type="box" size="0.018 0.018 {sh/2:.3f}"
            pos=" 0.17 -0.12 {sh/2:.3f}" rgba="0.45 0.35 0.20 1"
            contype="0" conaffinity="0"/>
    </body>

    <!-- ── Stage 3: Truck ───────────────────────────────────────────── -->
    <body name="truck_body" pos="{tx:.3f} {ty:.3f} 0.000">
      <geom name="truck_bed_floor" type="box" size="1.000 0.650 0.025"
            pos="0 0 {tbh:.3f}" rgba="0.45 0.45 0.50 1"
            contype="0" conaffinity="0"/>
      <geom name="truck_wall_back"  type="box" size="0.025 0.650 0.250"
            pos="-1.025 0 {tbh+0.25:.3f}" rgba="0.30 0.30 0.35 1"
            contype="0" conaffinity="0"/>
      <geom name="truck_wall_front" type="box" size="0.025 0.650 0.250"
            pos=" 1.025 0 {tbh+0.25:.3f}" rgba="0.30 0.30 0.35 1"
            contype="0" conaffinity="0"/>
      <geom name="truck_wall_left"  type="box" size="1.000 0.025 0.250"
            pos="0  0.675 {tbh+0.25:.3f}" rgba="0.30 0.30 0.35 1"
            contype="0" conaffinity="0"/>
      <geom name="truck_wall_right" type="box" size="1.000 0.025 0.250"
            pos="0 -0.675 {tbh+0.25:.3f}" rgba="0.30 0.30 0.35 1"
            contype="0" conaffinity="0"/>
      <geom name="truck_cabin"      type="box" size="0.350 0.500 0.400"
            pos="1.400 0 {tbh+0.175:.3f}" rgba="0.20 0.20 0.65 1"
            contype="0" conaffinity="0"/>
      <geom name="truck_wheel_fl" type="cylinder" size="0.220 0.120"
            pos=" 0.700  0.700 0.10" euler="1.5708 0 0"
            rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <geom name="truck_wheel_fr" type="cylinder" size="0.220 0.120"
            pos=" 0.700 -0.700 0.10" euler="1.5708 0 0"
            rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <geom name="truck_wheel_bl" type="cylinder" size="0.220 0.120"
            pos="-0.700  0.700 0.10" euler="1.5708 0 0"
            rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <geom name="truck_wheel_br" type="cylinder" size="0.220 0.120"
            pos="-0.700 -0.700 0.10" euler="1.5708 0 0"
            rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <inertial pos="0 0 0.3" mass="5000" diaginertia="50 50 100"/>
    </body>

    <!-- ── Stage 3: Package mocap bodies ────────────────────────────── -->
    <body name="s3_pkg_staged" mocap="true" pos="0 0 -10">
      <geom name="s3_pkg_staged_geom" type="box"
            size="{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}"
            rgba="0 0 0 0" contype="0" conaffinity="0"/>
    </body>
    <body name="s3_pkg_truck" mocap="true" pos="0 0 -10">
      <geom name="s3_pkg_truck_geom" type="box"
            size="{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}"
            rgba="0 0 0 0" contype="0" conaffinity="0"/>
    </body>

    <!-- ── Stage 3: Franka Panda arm #2 ─────────────────────────────── -->
    <body name="{PREFIX}panda_mount" pos="{bx:.4f} {by:.4f} {bz:.4f}">
      {prefixed['worldbody']}
    </body>
"""
        return {
            "worldbody": worldbody,
            "actuator":  prefixed["actuator"],
            "tendon":    prefixed["tendon"],
            "equality":  prefixed["equality"],
            "contact":   prefixed["contact"],
        }

    # ── Setup (call after MjModel created) ────────────────────────────────────

    def setup(self, model, data):
        def bid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        def gid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        def aid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

        self._ctrl_idx = [aid(f"{PREFIX}actuator{i+1}") for i in range(8)]
        self._hand_id  = bid(f"{PREFIX}hand")

        sb = bid("s3_pkg_staged");  self._staged_mid = model.body_mocapid[sb]
        tb = bid("s3_pkg_truck");   self._truck_mid  = model.body_mocapid[tb]
        self._staged_gid = gid("s3_pkg_staged_geom")
        self._truck_gid  = gid("s3_pkg_truck_geom")

        for i, q in enumerate(Q3_HOME):
            data.ctrl[self._ctrl_idx[i]] = q
        data.ctrl[self._ctrl_idx[7]] = GRIPPER_OPEN

        mujoco.mj_forward(model, data)
        print(f"[Stage3] Truck loading station ready  "
              f"(arm ctrl[{self._ctrl_idx[0]}..{self._ctrl_idx[7]}])")

    # ── Activation (call once when AMR arrives) ────────────────────────────────

    def activate(self, pkg_info: dict, data):
        """
        Enqueue a package for unloading and place it on the staging table.
        pkg_info: dict with keys item_name, mass_kg, rgba, pkg_type.
        """
        self._lifo.push(pkg_info)
        # Show staged package
        staged_z = STAGING_H + PKG_HALF[2]
        data.mocap_pos[self._staged_mid] = [STAGING_POS[0], STAGING_POS[1], staged_z]
        print(f"[Stage3] Package '{pkg_info['pkg_type']}' on staging table  "
              f"(LIFO depth={len(self._lifo)})")

    # ── Per-step state machine ─────────────────────────────────────────────────

    def step(self, model, data, step_count: int) -> bool:
        """
        Advance Stage 3 arm one simulation step.
        Returns True when all packages have been loaded into the truck.
        """
        self._state_step += 1
        s = self._state

        if s == S3ArmState.IDLE:
            if not self._lifo.is_empty():
                self._active_pkg = self._lifo.pop()
                rgba = self._active_pkg.get("rgba", [0.6, 0.4, 0.2, 1.0])
                # Colour both package mocap geoms with the package colour
                model.geom_rgba[self._staged_gid] = rgba
                model.geom_rgba[self._truck_gid]  = rgba
                print(f"[Stage3] Starting unload: '{self._active_pkg['pkg_type']}'  "
                      f"LIFO pop — BLB bin packing...")
                # Plan RRT from HOME to PRE_PICK (stored; we just set target_q directly)
                self._rrt_waypoints = self._rrt.plan(Q3_HOME, Q3_PRE_PICK)
                self._rrt_idx = 0
                self._state = S3ArmState.PRE_PICK
                self._state_step = 0

        elif s == S3ArmState.PRE_PICK:
            self._target_q = Q3_PRE_PICK
            self._gripper  = GRIPPER_OPEN
            if self._state_step >= STATE_HOLD["PRE_PICK"]:
                self._state = S3ArmState.PICK; self._state_step = 0

        elif s == S3ArmState.PICK:
            self._target_q = Q3_PICK
            if self._state_step >= STATE_HOLD["PICK"] // 2:
                self._gripper = GRIPPER_CLOSED
            if self._state_step >= STATE_HOLD["PICK"]:
                self._pkg_in_hand = True
                # Hide staged pkg — package now "in gripper"
                data.mocap_pos[self._staged_mid] = [0.0, 0.0, -10.0]
                self._state = S3ArmState.LIFT; self._state_step = 0
                print(f"[Stage3] Grasped '{self._active_pkg['pkg_type']}'")

        elif s == S3ArmState.LIFT:
            self._target_q = Q3_LIFT
            self._gripper  = GRIPPER_CLOSED
            self._track_pkg_to_hand(data)
            if self._state_step >= STATE_HOLD["LIFT"]:
                self._state = S3ArmState.TRANSPORT; self._state_step = 0

        elif s == S3ArmState.TRANSPORT:
            self._target_q = Q3_TRANSPORT
            self._gripper  = GRIPPER_CLOSED
            self._track_pkg_to_hand(data)
            if self._state_step >= STATE_HOLD["TRANSPORT"]:
                self._state = S3ArmState.DEPOSIT; self._state_step = 0

        elif s == S3ArmState.DEPOSIT:
            self._target_q = Q3_DEPOSIT
            self._gripper  = GRIPPER_CLOSED
            self._track_pkg_to_hand(data)
            if self._state_step >= STATE_HOLD["DEPOSIT"] // 2:
                self._gripper = GRIPPER_OPEN
            if self._state_step >= STATE_HOLD["DEPOSIT"]:
                self._pkg_in_hand = False
                # BLB: compute final resting position in truck
                item_full = PKG_HALF * 2
                truck_pos = self._packer.pack(item_full)
                if truck_pos is None:
                    # Fallback: stack in centre
                    truck_pos = np.array([TRUCK_POS[0],
                                          TRUCK_POS[1],
                                          TRUCK_BED_H + 0.025 + PKG_HALF[2]])
                data.mocap_pos[self._truck_mid] = truck_pos
                # Hide the hand-tracked pkg (truck mocap now shows it)
                data.mocap_pos[self._staged_mid] = [0.0, 0.0, -10.0]
                self._pkg_count += 1
                print(f"[Stage3] Package deposited into truck  "
                      f"pos=({truck_pos[0]:.2f},{truck_pos[1]:.2f},{truck_pos[2]:.2f})  "
                      f"total loaded={self._pkg_count}")
                self._state = S3ArmState.RETRACT; self._state_step = 0

        elif s == S3ArmState.RETRACT:
            self._target_q = Q3_HOME
            self._gripper  = GRIPPER_OPEN
            if self._state_step >= STATE_HOLD["RETRACT"]:
                if not self._lifo.is_empty():
                    # More packages → restart
                    self._state = S3ArmState.IDLE; self._state_step = 0
                else:
                    self._state = S3ArmState.DONE; self._state_step = 0
                    print("[Stage3] All packages loaded — arm at HOME")

        elif s == S3ArmState.DONE:
            self._target_q = Q3_HOME
            self._gripper  = GRIPPER_OPEN

        self._apply_ctrl(data)
        return self._state == S3ArmState.DONE

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_ee_pos(self, data) -> np.ndarray:
        pos = data.xpos[self._hand_id].copy()
        mat = data.xmat[self._hand_id].reshape(3, 3)
        return pos + mat @ np.array([0.0, 0.0, 0.17])

    def _track_pkg_to_hand(self, data):
        """Move s3_pkg_staged mocap to end-effector position."""
        data.mocap_pos[self._staged_mid] = self._get_ee_pos(data)

    def _apply_ctrl(self, data):
        for i, q in enumerate(self._target_q):
            data.ctrl[self._ctrl_idx[i]] = float(q)
        data.ctrl[self._ctrl_idx[7]] = float(self._gripper)

    def set_pkg_color(self, model, rgba):
        """Override staged and truck package geom colour."""
        model.geom_rgba[self._staged_gid] = rgba
        model.geom_rgba[self._truck_gid]  = rgba

    def get_packages_loaded(self) -> int:
        return self._pkg_count

    def get_status(self) -> str:
        active = self._active_pkg["pkg_type"] if self._active_pkg else "—"
        return (f"[Stage3] state={self._state.name:10s}  "
                f"pkg={active:10s}  loaded={self._pkg_count}")

    # ── Standalone run ────────────────────────────────────────────────────────

    def run(self, steps: int = 2800):
        """Run Stage 3 in standalone mode (single Panda arm, staging table, truck)."""
        prefixed = _prefix_panda_xml(PREFIX)
        assets_rel = str(PANDA_DIR / "assets").replace("\\", "/")

        # Parse panda defaults + asset section from original (un-prefixed) tree
        orig_tree = ET.parse(str(PANDA_XML))
        orig_root = orig_tree.getroot()
        panda_default = ET.tostring(orig_root.find("default"), encoding="unicode")
        panda_assets  = _elem_inner_xml(orig_root.find("asset"))

        snip = self.get_xml_snippets()
        p    = PKG_HALF

        xml = f"""\
<mujoco model="stage3_standalone">
  <compiler angle="radian" meshdir="{assets_rel}" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.005" integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4" specular="0 0 0"/>
  </visual>

  {panda_default}

  <asset>
    {panda_assets}
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.85 0.85 0.90" rgb2="0.70 0.70 0.75"/>
    <material name="floor_mat" texture="grid" texrepeat="6 6" reflectance="0.15"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="12 12 0.1"
          material="floor_mat" contype="0" conaffinity="0" pos="4 4 0"/>
    <light pos="7.5 7.0 4.0" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.1 0.1 0.1"/>
    {snip['worldbody']}
  </worldbody>

  <tendon>
    {snip['tendon']}
  </tendon>
  <equality>
    {snip['equality']}
  </equality>
  {'<contact>' + snip['contact'] + '</contact>' if snip['contact'].strip() else ''}

  <actuator>
    {snip['actuator']}
  </actuator>
</mujoco>
"""
        output = str(pathlib.Path(__file__).parent / "stage3_standalone.xml")
        with open(output, "w", encoding="utf-8") as f:
            f.write(xml)

        model = mujoco.MjModel.from_xml_path(output)
        data  = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        self.setup(model, data)

        # Simulate a package arriving from AMR
        fake_pkg = {
            "pkg_type":  "standard",
            "item_name": "medium_box",
            "mass_kg":   15.0,
            "rgba":      [0.25, 0.55, 0.90, 1.0],
        }
        self.activate(fake_pkg, data)

        print("\n" + "="*58)
        print("  STAGE 3 — TRUCK LOADING STATION  (standalone)")
        print("="*58)
        print(f"  Robot    : Franka Panda #2 (s3_ prefix)")
        print(f"  Arm base : {ARM3_BASE}")
        print(f"  Staging  : ({STAGING_POS[0]}, {STAGING_POS[1]})  h={STAGING_H}m")
        print(f"  Truck    : {TRUCK_POS}")
        print(f"  Planner  : RRT (step={self._rrt.step_size}, iter={self._rrt.max_iter})")
        print(f"  Packing  : BLB 3D bin packer")
        print("="*58 + "\n")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [8.0, 7.0, 0.8]
            viewer.cam.distance  = 4.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth   = 210

            step = 0
            while viewer.is_running() and step < steps:
                done = self.step(model, data, step)
                if step % 500 == 0 and step > 0:
                    print(f"  t={data.time:6.1f}s  {self.get_status()}")
                mujoco.mj_step(model, data)
                viewer.sync()
                step += 1
                if done and step > 200:
                    print(f"\n[Stage3] Complete — {self._pkg_count} pkg(s) in truck.")
                    break

        print(f"\n[Stage3] Standalone closed.  Packages loaded: {self._pkg_count}")


# ── Standalone runner ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ts = TruckLoadingStation()
    ts.run(steps=2800)
