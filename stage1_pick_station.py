"""
Stage 1 — Source Pick Station
Franka Emika Panda (DeepMind Menagerie) + Simulated YOLOv8 + Conveyor Belt

Uses the real Menagerie mesh model from franka_emika_panda/panda.xml.
Packages arrive on a conveyor belt, YOLOv8 detection prioritises them,
arm picks in order and loads onto the waiting AMR.

Run standalone:  python stage1_pick_station.py
Import:          from stage1_pick_station import PickStation
"""

import os
import pathlib
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import mujoco.viewer
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

PANDA_DIR = pathlib.Path(__file__).parent / "franka_emika_panda"
PANDA_XML = PANDA_DIR / "panda.xml"

# ── Package catalogue  (priority, mass_kg, rgba, amr_item_name) ───────────────
PACKAGE_CATALOGUE = {
    "fragile":  (10, 4.0,  [0.95, 0.25, 0.25, 1.0], "small_box"),
    "express":  (8,  8.0,  [0.95, 0.60, 0.10, 1.0], "small_box"),
    "standard": (4,  15.0, [0.25, 0.55, 0.90, 1.0], "medium_box"),
    "bulk":     (2,  20.0, [0.35, 0.75, 0.30, 1.0], "medium_box"),
}

# ── Simulated YOLOv8 ──────────────────────────────────────────────────────────
class YOLOv8Detector:
    """
    Simulates YOLOv8 inference on the pick-station camera feed.
    Replace detect() body with ultralytics YOLO inference on rendered frames
    to run the real model.
    """
    CONF_THRESHOLD = 0.45

    def __init__(self):
        self._rng = np.random.default_rng(7)

    def detect(self, packages: list) -> list:
        detections = []
        for pkg in packages:
            if pkg.picked or pkg.detected:
                continue
            conf = float(np.clip(0.72 + self._rng.uniform(-0.08, 0.22), 0, 0.99))
            if conf < self.CONF_THRESHOLD:
                continue
            noise = self._rng.uniform(-2, 2, 4)
            cat   = PACKAGE_CATALOGUE[pkg.pkg_type]
            detections.append({
                "id":       pkg.id,
                "class":    pkg.pkg_type,
                "priority": cat[0],
                "mass_kg":  cat[1],
                "rgba":     cat[2],
                "amr_item": cat[3],
                "conf":     round(conf, 3),
                "bbox":     [round(320 + noise[0], 1), round(240 + noise[1], 1),
                             round(80  + noise[2], 1), round(60  + noise[3], 1)],
                "package":  pkg,
            })
        detections.sort(key=lambda d: d["priority"], reverse=True)
        return detections


# ── Package data class ─────────────────────────────────────────────────────────
@dataclass
class Package:
    id:       int
    pkg_type: str
    belt_x:   float
    belt_y:   float
    belt_z:   float
    detected: bool = False
    picked:   bool = False
    loaded:   bool = False


# ── Arm state machine ──────────────────────────────────────────────────────────
class ArmState(Enum):
    IDLE      = auto()
    DETECT    = auto()
    PRE_GRASP = auto()
    GRASP     = auto()
    LIFT      = auto()
    TRANSFER  = auto()
    RELEASE   = auto()
    RETRACT   = auto()
    DONE      = auto()

# ── Joint keyframes (radians) — Panda convention ──────────────────────────────
# ARM_BASE = (0.3, 0.0, 0.0); conveyor is at -Y, load zone is at +Y
# joint1 > 0  → arm swings toward +Y  (load zone side)
# joint1 < 0  → arm swings toward -Y  (conveyor side)
Q_HOME      = np.array([ 0.000, -0.785,  0.000, -2.356,  0.000,  1.571,  0.785])
Q_PRE_GRASP = np.array([-1.500,  0.400,  0.100, -1.600,  0.000,  2.000,  0.500])
Q_GRASP     = np.array([-1.500,  0.620,  0.100, -1.300,  0.000,  1.900,  0.500])
Q_LIFT      = np.array([-1.500,  0.050,  0.100, -2.300,  0.000,  2.300,  0.500])
Q_TRANSFER  = np.array([ 1.450,  0.200,  0.000, -2.100,  0.000,  2.300,  0.700])
Q_RELEASE   = np.array([ 1.450,  0.500,  0.000, -1.800,  0.000,  2.100,  0.700])

# Gripper ctrl (actuator8 in panda.xml, mapped 0-255: 255=open, 0=closed)
GRIPPER_OPEN   = 255.0
GRIPPER_CLOSED =   5.0

# Steps at 200 Hz to hold each state
STATE_HOLD = {
    ArmState.IDLE:      0,
    ArmState.DETECT:    180,
    ArmState.PRE_GRASP: 520,
    ArmState.GRASP:     340,
    ArmState.LIFT:      420,
    ArmState.TRANSFER:  560,
    ArmState.RELEASE:   300,
    ArmState.RETRACT:   520,
    ArmState.DONE:      0,
}


# ── Conveyor + package XML fragments ──────────────────────────────────────────
def _conveyor_worldbody_xml(cx, cy, cz):
    leg_h = cz / 2
    return f"""
    <!-- Conveyor belt -->
    <body name="conveyor_body" pos="{cx:.3f} {cy:.3f} {cz:.3f}">
      <geom name="conv_belt"    type="box" size="0.55 0.13 0.018"
            rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>
      <geom name="conv_s0" type="box" size="0.04 0.13 0.021"
            pos="-0.40 0 0" rgba="0.90 0.80 0.10 1" contype="0" conaffinity="0"/>
      <geom name="conv_s1" type="box" size="0.04 0.13 0.021"
            pos="-0.15 0 0" rgba="0.90 0.80 0.10 1" contype="0" conaffinity="0"/>
      <geom name="conv_s2" type="box" size="0.04 0.13 0.021"
            pos=" 0.10 0 0" rgba="0.90 0.80 0.10 1" contype="0" conaffinity="0"/>
      <geom name="conv_s3" type="box" size="0.04 0.13 0.021"
            pos=" 0.35 0 0" rgba="0.90 0.80 0.10 1" contype="0" conaffinity="0"/>
      <geom name="conv_rl" type="box" size="0.56 0.016 0.045"
            pos="0  0.146 0.030" rgba="0.42 0.42 0.42 1" contype="0" conaffinity="0"/>
      <geom name="conv_rr" type="box" size="0.56 0.016 0.045"
            pos="0 -0.146 0.030" rgba="0.42 0.42 0.42 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.025 0.025 {leg_h:.3f}"
            pos="-0.50  0.11 {-leg_h:.3f}" rgba="0.28 0.28 0.28 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.025 0.025 {leg_h:.3f}"
            pos="-0.50 -0.11 {-leg_h:.3f}" rgba="0.28 0.28 0.28 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.025 0.025 {leg_h:.3f}"
            pos=" 0.50  0.11 {-leg_h:.3f}" rgba="0.28 0.28 0.28 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.025 0.025 {leg_h:.3f}"
            pos=" 0.50 -0.11 {-leg_h:.3f}" rgba="0.28 0.28 0.28 1" contype="0" conaffinity="0"/>
    </body>
    <!-- Package mocap bodies (kinematically controlled) -->
    <body name="pkg_body_0" mocap="true" pos="0 0 -10">
      <geom name="pkg_geom_0" type="box" size="0.085 0.085 0.075"
            rgba="0.5 0.5 0.5 0" contype="0" conaffinity="0"/>
    </body>
    <body name="pkg_body_1" mocap="true" pos="0 0 -10">
      <geom name="pkg_geom_1" type="box" size="0.085 0.085 0.075"
            rgba="0.5 0.5 0.5 0" contype="0" conaffinity="0"/>
    </body>
    <body name="pkg_body_2" mocap="true" pos="0 0 -10">
      <geom name="pkg_geom_2" type="box" size="0.085 0.085 0.075"
            rgba="0.5 0.5 0.5 0" contype="0" conaffinity="0"/>
    </body>
    <body name="pkg_body_3" mocap="true" pos="0 0 -10">
      <geom name="pkg_geom_3" type="box" size="0.085 0.085 0.075"
            rgba="0.5 0.5 0.5 0" contype="0" conaffinity="0"/>
    </body>
"""


# ── XML builder (parses real panda.xml) ───────────────────────────────────────
def _elem_inner_xml(elem) -> str:
    """Return XML string of all children of elem (not elem itself)."""
    parts = []
    if elem.text and elem.text.strip():
        parts.append(elem.text.strip())
    for child in elem:
        parts.append(ET.tostring(child, encoding='unicode'))
    return "\n    ".join(parts)


def build_combined_xml_file(arm_base_pos: tuple,
                             conveyor_pos: tuple,
                             extra_worldbody: str = "",
                             extra_actuators: str = "",
                             extra_tendon_inner: str = "",
                             extra_equality_inner: str = "",
                             extra_contact_inner: str = "",
                             output_filename: str = "combined_scene.xml",
                             load_zone: tuple = (0.45, 0.50)) -> str:
    """
    Parse the real Menagerie panda.xml, merge with warehouse+AMR+Stage3 XML,
    write combined_scene.xml to the warehouse directory.
    Returns absolute path to the written file.
    """
    output_path = str(pathlib.Path(__file__).parent / output_filename)
    assets_rel  = os.path.relpath(str(PANDA_DIR / "assets"),
                                   str(pathlib.Path(__file__).parent))
    # Normalise to forward slashes (MuJoCo on Windows prefers them)
    assets_rel  = assets_rel.replace("\\", "/")

    # Parse panda.xml
    tree = ET.parse(str(PANDA_XML))
    root = tree.getroot()

    # Extract sections
    panda_default  = ET.tostring(root.find("default"),  encoding="unicode")
    panda_assets   = _elem_inner_xml(root.find("asset"))
    panda_actuator = _elem_inner_xml(root.find("actuator"))

    # Extract inner XML so we can merge with stage3's tendon/equality/contact
    panda_tendon_inner   = _elem_inner_xml(root.find("tendon")) \
                           if root.find("tendon")   is not None else ""
    panda_equality_inner = _elem_inner_xml(root.find("equality")) \
                           if root.find("equality") is not None else ""
    panda_contact_inner  = _elem_inner_xml(root.find("contact")) \
                           if root.find("contact")  is not None else ""

    # Worldbody: strip light, extract inner (link0 body)
    wb = root.find("worldbody")
    for light in wb.findall("light"):
        wb.remove(light)
    panda_wb_inner = _elem_inner_xml(wb)   # this is the link0 body XML

    bx, by, bz    = arm_base_pos
    cx, cy, cz    = conveyor_pos
    lzx, lzy      = load_zone

    # Merge tendon / equality / contact inner XML from panda1 + stage3
    merged_tendon   = panda_tendon_inner   + ("\n    " + extra_tendon_inner   if extra_tendon_inner.strip()   else "")
    merged_equality = panda_equality_inner + ("\n    " + extra_equality_inner if extra_equality_inner.strip() else "")
    merged_contact  = panda_contact_inner  + ("\n    " + extra_contact_inner  if extra_contact_inner.strip()  else "")

    combined = f"""\
<mujoco model="warehouse_integrated">
  <compiler angle="radian" meshdir="{assets_rel}" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.005" integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <rgba haze="0.8 0.9 1.0 1"/>
  </visual>

  <!-- Panda defaults (class="panda", shared by both arms) -->
  {panda_default}

  <asset>
    <!-- Panda mesh assets (shared by both arms) -->
    {panda_assets}

    <!-- Warehouse materials -->
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.85 0.85 0.90" rgb2="0.70 0.70 0.75"/>
    <material name="floor_mat"  texture="grid" texrepeat="6 6" reflectance="0.15"/>
    <material name="body_mat"   rgba="0.20 0.55 0.85 1"/>
    <material name="wheel_mat"  rgba="0.15 0.15 0.15 1"/>
    <material name="trail_mat"  rgba="1.00 0.65 0.10 0.80"/>
    <material name="obs_r_mat"  rgba="0.75 0.28 0.18 0.90"/>
    <material name="obs_g_mat"  rgba="0.22 0.62 0.35 0.90"/>
    <material name="obs_b_mat"  rgba="0.20 0.40 0.90 0.90"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="12 12 0.1"
          material="floor_mat" contype="0" conaffinity="0" pos="4 4 0"/>
    <geom name="start_marker" type="cylinder" size="0.12 0.01" pos="0 0 0.01"
          material="trail_mat" contype="0" conaffinity="0"/>

    <!-- Load zone marker (where AMR parks to receive package) -->
    <geom name="load_zone_marker" type="cylinder" size="0.22 0.008"
          pos="{lzx:.2f} {lzy:.2f} 0.01" rgba="0.10 0.90 0.30 0.55"
          contype="0" conaffinity="0"/>

    <!-- Delivery goal marker -->
    <body name="goal_marker" mocap="true" pos="7 7 0">
      <geom type="cylinder" size="0.20 0.012"
            rgba="0.10 0.90 0.30 0.85" contype="0" conaffinity="0"/>
    </body>

    <!-- Warehouse obstacles -->
    <geom name="obs_d"    type="box"      size="0.35 0.20 0.35"
          pos="0.6 1.2 0.35"   material="obs_g_mat" contype="2" conaffinity="0"/>
    <geom name="obs_f"    type="cylinder" size="0.22 0.35"
          pos="1.5 0.7 0.35"   material="obs_b_mat" contype="2" conaffinity="0"/>
    <geom name="obs_b"    type="box"      size="0.25 0.40 0.50"
          pos="3.5 1.2 0.50"   material="obs_g_mat" contype="2" conaffinity="0"/>
    <geom name="obs_a"    type="box"      size="0.30 0.30 0.45"
          pos="0.8 2.5 0.45"   material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_test" type="box"      size="0.25 0.25 0.50"
          pos="2.2 2.0 0.50"   material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_e"    type="box"      size="0.22 0.25 0.40"
          pos="3.5 3.0 0.40"   material="obs_b_mat" contype="2" conaffinity="0"/>
    <geom name="obs_c"    type="cylinder" size="0.20 0.40"
          pos="2.0 3.8 0.40"   material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_g_1"  type="box"      size="0.25 0.22 0.38"
          pos="3.3 3.8 0.38"   material="obs_b_mat" contype="2" conaffinity="0"/>
    <geom name="obs_h"    type="box"      size="0.30 0.35 0.55"
          pos="5.5 0.6 0.55"   material="obs_g_mat" contype="2" conaffinity="0"/>
    <geom name="obs_m"    type="box"      size="0.28 0.22 0.48"
          pos="4.5 2.5 0.48"   material="obs_b_mat" contype="2" conaffinity="0"/>
    <geom name="obs_p"    type="cylinder" size="0.24 0.52"
          pos="6.5 2.0 0.52"   material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_i"    type="box"      size="0.32 0.28 0.62"
          pos="0.5 5.0 0.62"   material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_j"    type="cylinder" size="0.22 0.48"
          pos="0.7 6.8 0.48"   material="obs_b_mat" contype="2" conaffinity="0"/>
    <geom name="obs_q"    type="box"      size="0.38 0.25 0.42"
          pos="1.5 7.5 0.42"   material="obs_g_mat" contype="2" conaffinity="0"/>
    <geom name="obs_n"    type="box"      size="0.30 0.35 0.58"
          pos="2.5 5.2 0.58"   material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_o"    type="cylinder" size="0.26 0.50"
          pos="4.2 4.8 0.50"   material="obs_g_mat" contype="2" conaffinity="0"/>
    <geom name="obs_k"    type="box"      size="0.40 0.30 0.65"
          pos="6.0 5.5 0.65"   material="obs_r_mat" contype="2" conaffinity="0"/>
    <geom name="obs_l"    type="cylinder" size="0.22 0.44"
          pos="5.5 6.8 0.44"   material="obs_g_mat" contype="2" conaffinity="0"/>
    <geom name="obs_r_1"  type="box"      size="0.28 0.32 0.50"
          pos="7.0 4.5 0.50"   material="obs_b_mat" contype="2" conaffinity="0"/>

    <!-- Franka Panda arm #1 — Source Pick Station -->
    <body name="panda_mount" pos="{bx:.4f} {by:.4f} {bz:.4f}">
      {panda_wb_inner}
    </body>

    <!-- Conveyor belt + packages -->
    {_conveyor_worldbody_xml(cx, cy, cz)}

    <!-- AMR robot + Stage 3 elements -->
    {extra_worldbody}

  </worldbody>

  <tendon>
    {merged_tendon}
  </tendon>
  <equality>
    {merged_equality}
  </equality>
  {'<contact>' + chr(10) + '    ' + merged_contact + chr(10) + '  </contact>' if merged_contact.strip() else ''}

  <actuator>
    <!-- Panda #1 actuators (indices 0-7) -->
    {panda_actuator}
    <!-- AMR + Stage 3 actuators (indices 8+) -->
    {extra_actuators}
  </actuator>
</mujoco>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)
    return output_path


# ── PickStation class ──────────────────────────────────────────────────────────
class PickStation:
    """
    Manages the Franka Panda pick station using the real Menagerie model.

    Integrated usage
    ────────────────
        ps = PickStation(base_pos=(0, -1.0, 0), conveyor_offset=(0, -0.7, 0.70))
        xml_path = ps.build_combined_xml(amr_worldbody_xml, amr_actuator_xml)
        model = mujoco.MjModel.from_xml_path(xml_path)
        data  = mujoco.MjData(model)
        ps.setup(model, data)
        # each step:
        pkg = ps.step(model, data, step)   # returns dict when pkg ready for AMR
    """

    N_PACKAGES = 4

    # Panda actuator ctrl indices (order from panda.xml: actuator1-8)
    CTRL_J1, CTRL_J2, CTRL_J3, CTRL_J4 = 0, 1, 2, 3
    CTRL_J5, CTRL_J6, CTRL_J7          = 4, 5, 6
    CTRL_GRIP                           = 7   # gripper tendon (0=closed, 255=open)

    def __init__(self,
                 base_pos: tuple = (0.3, 0.0, 0.0),
                 conveyor_offset: tuple = (0.0, -0.45, 0.65)):
        self._base_pos = base_pos
        bx, by, bz = base_pos
        cx = bx + conveyor_offset[0]
        cy = by + conveyor_offset[1]
        cz = conveyor_offset[2]
        self._conv_pos = (cx, cy, cz)

        # Package positions spread across conveyor
        self._belt_positions = [
            np.array([cx - 0.30 + i * 0.22, cy, cz + 0.09])
            for i in range(self.N_PACKAGES)
        ]

        # Deposit position: arm releases package here — kept close to the AMR
        # cargo bay height (base 0.18 + cargo offset 0.22 = 0.40 m above ground)
        self._deposit_pos = np.array([bx + 0.15, by + 0.50, bz + 0.42])

        # Populated by setup()
        self._hand_id    = -1
        self._pkg_mocap  = []
        self._pkg_geom   = []

        self._packages   = self._spawn_packages()
        self._detector   = YOLOv8Detector()
        self._state      = ArmState.IDLE
        self._state_step = 0
        self._target_q   = Q_HOME.copy()
        self._gripper    = GRIPPER_OPEN
        self._active_pkg: Optional[dict] = None
        self._pkg_count  = 0

    # ── XML helpers ───────────────────────────────────────────────────────────

    def build_combined_xml(self,
                            amr_worldbody_xml: str = "",
                            amr_actuator_xml: str = "",
                            extra_tendon_inner: str = "",
                            extra_equality_inner: str = "",
                            extra_contact_inner: str = "",
                            output_filename: str = "combined_scene.xml",
                            load_zone: tuple = (0.45, 0.50)) -> str:
        """Write combined XML to disk and return the file path."""
        return build_combined_xml_file(
            arm_base_pos=self._base_pos,
            conveyor_pos=self._conv_pos,
            extra_worldbody=amr_worldbody_xml,
            extra_actuators=amr_actuator_xml,
            extra_tendon_inner=extra_tendon_inner,
            extra_equality_inner=extra_equality_inner,
            extra_contact_inner=extra_contact_inner,
            output_filename=output_filename,
            load_zone=load_zone,
        )

    def build_standalone_xml(self) -> str:
        """Write standalone pick-station XML (no AMR) and return file path."""
        amr_wb = ""
        amr_act = ""
        return self.build_combined_xml(amr_wb, amr_act, "standalone_pick.xml")

    # ── Package management ────────────────────────────────────────────────────

    def _spawn_packages(self) -> List[Package]:
        types = list(PACKAGE_CATALOGUE.keys())
        return [
            Package(id=i, pkg_type=types[i],
                    belt_x=self._belt_positions[i][0],
                    belt_y=self._belt_positions[i][1],
                    belt_z=self._belt_positions[i][2])
            for i in range(self.N_PACKAGES)
        ]

    # ── Setup (call after MjModel created) ────────────────────────────────────

    def setup(self, model, data):
        def bid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        def gid(n): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)

        self._hand_id = bid("hand")

        self._pkg_mocap = []
        self._pkg_geom  = []
        for i in range(self.N_PACKAGES):
            mbody = bid(f"pkg_body_{i}")
            self._pkg_mocap.append(model.body_mocapid[mbody])
            self._pkg_geom.append(gid(f"pkg_geom_{i}"))

        # Place packages on belt
        for i, pkg in enumerate(self._packages):
            mid = self._pkg_mocap[i]
            data.mocap_pos[mid] = [pkg.belt_x, pkg.belt_y, pkg.belt_z]
            model.geom_rgba[self._pkg_geom[i]] = PACKAGE_CATALOGUE[pkg.pkg_type][2]

        # Initialise arm at home pose (from panda keyframe)
        data.ctrl[self.CTRL_J1]  = Q_HOME[0]
        data.ctrl[self.CTRL_J2]  = Q_HOME[1]
        data.ctrl[self.CTRL_J3]  = Q_HOME[2]
        data.ctrl[self.CTRL_J4]  = Q_HOME[3]
        data.ctrl[self.CTRL_J5]  = Q_HOME[4]
        data.ctrl[self.CTRL_J6]  = Q_HOME[5]
        data.ctrl[self.CTRL_J7]  = Q_HOME[6]
        data.ctrl[self.CTRL_GRIP] = GRIPPER_OPEN

        mujoco.mj_forward(model, data)
        print(f"[Stage1] Pick station ready — {self.N_PACKAGES} packages on belt")
        print(f"[Stage1] Arm base   : {self._base_pos}")
        print(f"[Stage1] Conveyor   : {self._conv_pos}")
        print(f"[Stage1] Deposit pos: {self._deposit_pos}")

    # ── Per-step state machine ─────────────────────────────────────────────────

    def step(self, model, data, step_count: int) -> Optional[dict]:
        """
        Advance pick station by one step.
        Returns payload dict when package is deposited and ready for AMR.
        """
        self._state_step += 1
        result = None
        s = self._state

        if s == ArmState.IDLE:
            if step_count > 100:
                self._state = ArmState.DETECT
                self._state_step = 0

        elif s == ArmState.DETECT:
            if self._state_step == 1:
                queue = self._detector.detect(self._packages)
                if queue:
                    for d in queue:
                        print(f"  [YOLOv8] '{d['class']}' conf={d['conf']:.2f} "
                              f"priority={d['priority']}  bbox={d['bbox']}")
                    self._active_pkg = queue[0]
                    self._active_pkg["package"].detected = True
                    print(f"  [Stage1] Picking '{self._active_pkg['class']}' "
                          f"(priority {self._active_pkg['priority']})")
                    self._state = ArmState.PRE_GRASP
                    self._state_step = 0

        elif s == ArmState.PRE_GRASP:
            self._target_q = Q_PRE_GRASP
            self._gripper  = GRIPPER_OPEN
            if self._state_step >= STATE_HOLD[s]:
                self._state = ArmState.GRASP; self._state_step = 0

        elif s == ArmState.GRASP:
            self._target_q = Q_GRASP
            if self._state_step >= STATE_HOLD[s] // 2:
                self._gripper = GRIPPER_CLOSED
            if self._state_step >= STATE_HOLD[s]:
                self._active_pkg["package"].picked = True
                self._state = ArmState.LIFT; self._state_step = 0
                print(f"  [Stage1] Grasped '{self._active_pkg['class']}'")

        elif s == ArmState.LIFT:
            self._target_q = Q_LIFT
            self._gripper  = GRIPPER_CLOSED
            self._track_pkg_to_hand(data)
            if self._state_step >= STATE_HOLD[s]:
                self._state = ArmState.TRANSFER; self._state_step = 0

        elif s == ArmState.TRANSFER:
            self._target_q = Q_TRANSFER
            self._gripper  = GRIPPER_CLOSED
            self._track_pkg_to_hand(data)
            if self._state_step >= STATE_HOLD[s]:
                self._state = ArmState.RELEASE; self._state_step = 0

        elif s == ArmState.RELEASE:
            self._target_q = Q_RELEASE
            self._gripper  = GRIPPER_CLOSED
            self._track_pkg_to_hand(data)
            if self._state_step >= STATE_HOLD[s] // 2:
                self._gripper = GRIPPER_OPEN
            if self._state_step >= STATE_HOLD[s]:
                pkg = self._active_pkg["package"]
                pkg.picked = False
                pkg.loaded = True
                mid = self._pkg_mocap[pkg.id]
                data.mocap_pos[mid] = list(self._deposit_pos)
                result = {
                    "item_name": self._active_pkg["amr_item"],
                    "mass_kg":   self._active_pkg["mass_kg"],
                    "rgba":      self._active_pkg["rgba"],
                    "pkg_type":  self._active_pkg["class"],
                }
                self._pkg_count += 1
                print(f"  [Stage1] Deposited '{pkg.pkg_type}' on AMR "
                      f"({result['mass_kg']} kg)")
                self._state = ArmState.RETRACT; self._state_step = 0

        elif s == ArmState.RETRACT:
            self._target_q = Q_HOME
            self._gripper  = GRIPPER_OPEN
            if self._state_step >= STATE_HOLD[s]:
                self._state = ArmState.DONE; self._state_step = 0
                print("[Stage1] Arm retracted to HOME — cycle complete")

        elif s == ArmState.DONE:
            self._target_q = Q_HOME
            self._gripper  = GRIPPER_OPEN

        self._apply_ctrl(data)
        return result

    def _get_ee_pos(self, data) -> np.ndarray:
        """Approximate EE position: hand body origin + 17cm along hand's local Z."""
        pos = data.xpos[self._hand_id].copy()
        mat = data.xmat[self._hand_id].reshape(3, 3)
        return pos + mat @ np.array([0.0, 0.0, 0.17])

    def _track_pkg_to_hand(self, data):
        if self._active_pkg is None: return
        pkg = self._active_pkg["package"]
        if not pkg.picked: return
        mid = self._pkg_mocap[pkg.id]
        data.mocap_pos[mid] = self._get_ee_pos(data)

    def _apply_ctrl(self, data):
        for i, q in enumerate(self._target_q):
            data.ctrl[i] = q
        data.ctrl[self.CTRL_GRIP] = self._gripper

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> str:
        active = self._active_pkg["class"] if self._active_pkg else "—"
        return (f"[Stage1] state={self._state.name:12s}  "
                f"pkg={active:10s}  picked={self._pkg_count}")

    def is_done(self) -> bool:
        return self._state == ArmState.DONE

    def set_loaded_pkg_pos(self, data, pos3d):
        """Move the deposited package's mocap body to pos3d (called every step
        during AMR navigation so the package visually rides on the robot)."""
        if self._active_pkg is None:
            return
        pkg = self._active_pkg["package"]
        if not pkg.loaded:
            return
        mid = self._pkg_mocap[pkg.id]
        data.mocap_pos[mid] = pos3d

    def run_standalone(self, steps: int = 1000):
        """Run pick station in standalone mode (no integration)."""
        xml_path = self.build_standalone_xml()
        print(f"[Stage1] Standalone XML: {xml_path}")

        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        self.setup(model, data)

        print("\n" + "="*56)
        print("  STAGE 1 — SOURCE PICK STATION  (standalone)")
        print("="*56)
        print("  Robot  : Franka Emika Panda (Menagerie mesh model)")
        print("  Sensor : Simulated YOLOv8 detection")
        print("  Queue  : 4 packages (fragile→express→standard→bulk)")
        print("="*56 + "\n")

        step = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.2, -1.2, 0.5]
            viewer.cam.distance = 3.2
            viewer.cam.elevation = -22
            viewer.cam.azimuth = 160

            while viewer.is_running() and step < steps:
                result = self.step(model, data, step)
                if result:
                    print(f"\n  >> Package for AMR: {result['pkg_type']} "
                          f"{result['mass_kg']} kg  ('{result['item_name']}')\n")
                if step % 500 == 0 and step > 0:
                    print(f"  t={data.time:6.1f}s  {self.get_status()}")
                mujoco.mj_step(model, data)
                viewer.sync()
                step += 1

        print(f"\n[Stage1] Closed after {step} steps.")
        return True


# ── Standalone runner ──────────────────────────────────────────────────────────
def _standalone_main():
    ps      = PickStation(base_pos=(0.3, 0.0, 0.0),
                          conveyor_offset=(0.0, -0.45, 0.65))
    xml_path = ps.build_standalone_xml()
    print(f"[Stage1] Standalone XML written to: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    ps.setup(model, data)

    print("\n" + "="*56)
    print("  STAGE 1 — SOURCE PICK STATION  (standalone)")
    print("="*56)
    print("  Robot  : Franka Emika Panda (Menagerie mesh model)")
    print("  Sensor : Simulated YOLOv8 detection")
    print("  Queue  : 4 packages (fragile→express→standard→bulk)")
    print("="*56 + "\n")

    step = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.2, -1.2, 0.5]
        viewer.cam.distance  = 3.2
        viewer.cam.elevation = -22
        viewer.cam.azimuth   = 160

        while viewer.is_running():
            result = ps.step(model, data, step)
            if result:
                print(f"\n  >> Package for AMR: {result['pkg_type']} "
                      f"{result['mass_kg']} kg  ('{result['item_name']}')\n")
            if step % 500 == 0 and step > 0:
                print(f"  t={data.time:6.1f}s  {ps.get_status()}")
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1

    print("\n[Stage1] Closed.")


if __name__ == "__main__":
    _standalone_main()
