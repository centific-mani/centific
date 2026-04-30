"""
AMR Payload Management System — Physics-based
----------------------------------------------
Weight is REAL mass applied to the MuJoCo model body.
Max capacity is DERIVED from actuator kv and robot specs, not hardcoded.
Motor effort is READ from data.actuator_force every step.

Connection points expected from amr_simulation.py:
    payload.apply_to_model(model, data, cargo_body_id)   → after load/unload
    payload.read_motor_effort(data)                       → every sim step
    payload.cargo_rgba()                                  → for geom colour
    payload.status()                                      → for terminal print
"""

import numpy as np
import mujoco


# ── Physical constants (mirror the XML actuator / joint values) ────────────────
KV_SLIDE        = 120.0   # <velocity kv="120"/>  — force per (m/s) error
DAMPING_SLIDE   = 5.0     # <joint damping="5"/>  — N·s/m
ROBOT_BASE_MASS = 20.0    # kg  — robot chassis (set in <inertial mass="20">)
MAX_VEL         = 0.35    # m/s — from amr_simulation.py MAX_VEL

# ── Cargo box half-dimensions (must match XML geom size) ──────────────────────
CARGO_HALF = np.array([0.24, 0.18, 0.12])   # metres

# ── Max capacity derivation ────────────────────────────────────────────────────
# Each slide motor peak force = kv × MAX_VEL = 120 × 0.35 = 42 N
# Two motors act together (x and y), so resultant peak = 42 × √2 ≈ 59 N
# Safety margin 0.70 → safe resultant = 59 × 0.70 ≈ 41 N
# Minimum acceptable acceleration when fully loaded = 0.5 m/s²
#   →  M_total_max = F_safe / a_min = 41 / 0.5 = 82 kg
#   →  MAX_PAYLOAD = M_total_max − ROBOT_BASE_MASS = 82 − 20 = 62 kg
#
# (Damping contributes F_damp = 5 × 0.35 = 1.75 N per axis — negligible here)

_F_PEAK_SINGLE   = KV_SLIDE * MAX_VEL                      # 42 N per axis
_F_PEAK_COMBINED = _F_PEAK_SINGLE * np.sqrt(2)             # 59 N resultant
_SAFETY_MARGIN   = 0.70
_F_SAFE          = _F_PEAK_COMBINED * _SAFETY_MARGIN       # 41.4 N
_DESIGN_ACCEL    = 0.50                                     # m/s²
_M_TOTAL_MAX     = _F_SAFE / _DESIGN_ACCEL                 # 82.8 kg
MAX_CAPACITY_KG  = round(_M_TOTAL_MAX - ROBOT_BASE_MASS, 1)  # 62.8 kg


# ── Catalogue ─────────────────────────────────────────────────────────────────
# weight_kg is the PHYSICAL mass injected into model.body_mass at runtime.
CATALOGUE = {
    "small_box":     {"weight_kg":  8.0,  "label": "Small cardboard box"},
    "medium_box":    {"weight_kg": 20.0,  "label": "Medium warehouse box"},
    "large_box":     {"weight_kg": 40.0,  "label": "Large pallet box"},
    "heavy_part":    {"weight_kg": 60.0,  "label": "Heavy machinery part"},
    "overload_item": {"weight_kg": 80.0,  "label": "Oversized item (exceeds limit)"},
}


def _box_inertia(mass: float, half: np.ndarray) -> np.ndarray:
    """Diagonal inertia tensor for a uniform solid box (half-dims a,b,c)."""
    a, b, c = half
    return (mass / 3.0) * np.array([b*b + c*c, a*a + c*c, a*a + b*b])


class PayloadManager:
    """
    Physics-aware payload manager.

    Key differences from a fake manager:
    - load/unload calls apply_to_model() → model.body_mass changes → motors feel it
    - read_motor_effort() reads data.actuator_force each step → real physics measurement
    - MAX_CAPACITY_KG is computed from kv, safety margin, and design acceleration
    """

    def __init__(self, robot_name: str = "AMR-01"):
        self.robot_name          = robot_name
        self.current_weight      = 0.0        # kg currently on robot
        self.loaded_items        = []

        # Live physics readings (updated by read_motor_effort every step)
        self.measured_effort_n   = 0.0        # resultant motor force   [N]
        self.capacity_used_phys  = 0.0        # fraction of peak force  [0..1]
        self._effort_history     = np.zeros(200)   # rolling window for smoothing
        self._effort_head        = 0

    # ── Load / unload ──────────────────────────────────────────────────────────

    def load(self, item_name: str, model, data, cargo_body_id: int) -> tuple:
        """
        Load item → update current_weight → push new mass into MuJoCo model.
        Returns (success: bool, message: str).
        """
        if item_name not in CATALOGUE:
            return False, f"Unknown item '{item_name}'. Choose from: {list(CATALOGUE)}"

        item      = CATALOGUE[item_name]
        projected = self.current_weight + item["weight_kg"]

        if projected > MAX_CAPACITY_KG:
            return False, (
                f"REJECTED  '{item['label']}'  ({item['weight_kg']} kg)  →  "
                f"would total {projected:.1f} kg, exceeding physics limit "
                f"of {MAX_CAPACITY_KG:.1f} kg  "
                f"(derived from kv={KV_SLIDE}, safety={_SAFETY_MARGIN})"
            )

        self.current_weight = projected
        self.loaded_items.append(item_name)
        self.apply_to_model(model, data, cargo_body_id)

        return True, (
            f"Loaded  '{item['label']}'  ({item['weight_kg']} kg)  |  "
            f"Total mass on robot: {self.current_weight:.1f} / "
            f"{MAX_CAPACITY_KG:.1f} kg  ({self.load_percent():.0f}%)"
        )

    def unload(self, item_name: str, model, data, cargo_body_id: int) -> tuple:
        """Unload item → push reduced mass into MuJoCo model."""
        if item_name not in self.loaded_items:
            return False, f"'{item_name}' is not on this robot."

        item = CATALOGUE[item_name]
        self.loaded_items.remove(item_name)
        self.current_weight = max(0.0, self.current_weight - item["weight_kg"])
        self.apply_to_model(model, data, cargo_body_id)

        return True, (
            f"Unloaded '{item['label']}'  ({item['weight_kg']} kg)  |  "
            f"Remaining: {self.current_weight:.1f} kg"
        )

    def unload_all(self, model, data, cargo_body_id: int) -> str:
        self.loaded_items   = []
        self.current_weight = 0.0
        self.apply_to_model(model, data, cargo_body_id)
        return "All cargo unloaded. Robot body mass reset to zero payload."

    # ── Physics interface ──────────────────────────────────────────────────────

    def apply_to_model(self, model, data, cargo_body_id: int):
        """
        Write current_weight into model.body_mass and recompute inertia.
        MuJoCo's solver uses this from the very next mj_step.
        """
        m = max(self.current_weight, 1e-4)          # avoid exactly-zero mass
        model.body_mass[cargo_body_id]      = m
        model.body_inertia[cargo_body_id]   = _box_inertia(m, CARGO_HALF)
        mujoco.mj_setConst(model, data)             # recompute mass-matrix constants

    def read_motor_effort(self, data):
        """
        Read actual actuator forces from data.actuator_force (set by MuJoCo solver).
        Smoothed over a rolling 200-step window to reduce noise.
        Call this every simulation step.
        """
        fx = float(data.actuator_force[0])          # x-slide motor
        fy = float(data.actuator_force[1])          # y-slide motor
        raw_effort = np.sqrt(fx*fx + fy*fy)

        # Rolling average
        self._effort_history[self._effort_head % 200] = raw_effort
        self._effort_head += 1
        self.measured_effort_n  = float(np.mean(self._effort_history))
        self.capacity_used_phys = self.measured_effort_n / _F_PEAK_COMBINED

    # ── Queries ────────────────────────────────────────────────────────────────

    def load_percent(self) -> float:
        return (self.current_weight / MAX_CAPACITY_KG) * 100.0

    def can_add(self, item_name: str) -> bool:
        if item_name not in CATALOGUE:
            return False
        return self.current_weight + CATALOGUE[item_name]["weight_kg"] <= MAX_CAPACITY_KG

    def state_label(self) -> str:
        pct = self.load_percent()
        if self.current_weight == 0.0: return "EMPTY"
        if pct > 100.0:                return "OVERLOADED"
        if pct >= 75.0:                return "NEAR LIMIT"
        return "OK"

    # ── Reporting ──────────────────────────────────────────────────────────────

    def status(self) -> str:
        pct     = self.load_percent()
        bar_len = 24
        filled  = int(min(pct, 100.0) / 100.0 * bar_len)
        bar     = "█" * filled + "░" * (bar_len - filled)

        items_str = (
            "\n  │    • " + "\n  │    • ".join(
                f"{n}  ({CATALOGUE[n]['weight_kg']} kg)  "
                f"— {CATALOGUE[n]['label']}"
                for n in self.loaded_items
            ) if self.loaded_items else "  none"
        )

        return (
            f"\n  ┌─── {self.robot_name}  Payload (Physics-based) "
            f"{'─'*22}┐\n"
            f"  │  Declared weight : {self.current_weight:6.1f} kg "
            f"(injected into model.body_mass)\n"
            f"  │  Capacity limit  : {MAX_CAPACITY_KG:6.1f} kg "
            f"(derived: kv={KV_SLIDE}, η={_SAFETY_MARGIN}, "
            f"a_min={_DESIGN_ACCEL} m/s²)\n"
            f"  │  Load fraction   : [{bar}] {pct:5.1f}%\n"
            f"  │  Motor effort    : {self.measured_effort_n:5.2f} N "
            f"(peak={_F_PEAK_COMBINED:.1f} N, "
            f"used={self.capacity_used_phys*100:.1f}%)\n"
            f"  │  State           : {self.state_label()}\n"
            f"  │  Cargo           :{items_str}\n"
            f"  └{'─'*52}┘\n"
        )

    def capacity_derivation(self) -> str:
        """Explain exactly how MAX_CAPACITY_KG was calculated."""
        return (
            f"\n  How MAX_CAPACITY_KG = {MAX_CAPACITY_KG:.1f} kg was derived:\n"
            f"    kv (slide actuator)     = {KV_SLIDE} N·s/m   [from XML kv=\"120\"]\n"
            f"    MAX_VEL                 = {MAX_VEL} m/s      [from simulation config]\n"
            f"    Peak force (one axis)   = kv × v = {_F_PEAK_SINGLE:.1f} N\n"
            f"    Peak resultant (x+y)    = {_F_PEAK_SINGLE:.1f} × √2 = "
            f"{_F_PEAK_COMBINED:.1f} N\n"
            f"    Safety margin           = {_SAFETY_MARGIN} "
            f"→ safe force = {_F_SAFE:.1f} N\n"
            f"    Min design acceleration = {_DESIGN_ACCEL} m/s²\n"
            f"    Max total mass          = {_F_SAFE:.1f} / {_DESIGN_ACCEL} = "
            f"{_M_TOTAL_MAX:.1f} kg\n"
            f"    Robot base mass         = {ROBOT_BASE_MASS:.1f} kg  "
            f"[from XML <inertial mass=\"20\">]\n"
            f"    MAX PAYLOAD             = {_M_TOTAL_MAX:.1f} − {ROBOT_BASE_MASS:.1f} "
            f"= {MAX_CAPACITY_KG:.1f} kg\n"
        )

    def catalogue_summary(self) -> str:
        lines = [
            f"\n  Warehouse catalogue  "
            f"(capacity = {MAX_CAPACITY_KG:.1f} kg, "
            f"headroom = {MAX_CAPACITY_KG - self.current_weight:.1f} kg):"
        ]
        for name, item in CATALOGUE.items():
            feasible = "✓" if self.can_add(name) else "✗ too heavy"
            lines.append(
                f"    {name:<16}  {item['weight_kg']:5.1f} kg  "
                f"— {item['label']:<32}  [{feasible}]"
            )
        return "\n".join(lines) + "\n"

    def cargo_rgba(self) -> list:
        """Colour the cargo geom by load state. Invisible when empty."""
        if self.current_weight < 1e-3:
            return [0.0, 0.0, 0.0, 0.0]
        pct = self.load_percent()
        if pct < 60.0:   return [0.20, 0.78, 0.25, 0.90]   # green
        if pct < 90.0:   return [0.95, 0.55, 0.05, 0.90]   # orange
        return             [0.85, 0.12, 0.10, 0.90]         # red
