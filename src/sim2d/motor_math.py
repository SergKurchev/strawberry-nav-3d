"""
Motor command limiting.

We intentionally reuse the *exact* math from:
  src_motors_gamepad/motors/runbot_motion_control/scripts/vel_acc_constraint.py

We load it directly from file so sim can run without ROS.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import importlib.util
import numpy as np


@lru_cache(maxsize=1)
def _vel_acc_constraint_mod():
    repo_root = Path(__file__).resolve().parents[2]  # .../strawberry-nav-3d
    p = repo_root / "src_motors_gamepad/motors/runbot_motion_control/scripts/vel_acc_constraint.py"
    if not p.exists():
        raise FileNotFoundError(f"vel_acc_constraint.py not found: {p}")

    spec = importlib.util.spec_from_file_location("vel_acc_constraint_ext", str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {p}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, "restrict_vel_acc"):
        raise ImportError("Loaded vel_acc_constraint has no restrict_vel_acc()")
    return mod


def restrict_vel_acc(target_vel: np.ndarray, prev_vel: np.ndarray, max_vel: np.ndarray, max_acc: np.ndarray) -> np.ndarray:
    """Thin wrapper around runbot_motion_control's restrict_vel_acc."""
    mod = _vel_acc_constraint_mod()
    return mod.restrict_vel_acc(target_vel, prev_vel, max_vel, max_acc)


def limit_vw(v_cmd: float, w_cmd: float, v_prev: float, w_prev: float, dt: float, cfg) -> tuple[float, float]:
    """
    Motor-like limiting of (v,w) using runbot_motion_control's restrict_vel_acc().

    Mirrors motors_node_x4 logic:
      max_acc is scaled by dt.

    Required cfg fields:
      MAX_V, MAX_A
    Optional:
      MAX_W, MAX_W_ACC
    """
    max_v = float(getattr(cfg, "MAX_V", 0.4))
    max_w = float(getattr(cfg, "MAX_W", 0.8))
    max_a = float(getattr(cfg, "MAX_A", 0.8))
    max_aw = float(getattr(cfg, "MAX_W_ACC", 1.6))

    target = np.array([v_cmd, w_cmd], dtype=np.float32)
    prev = np.array([v_prev, w_prev], dtype=np.float32)
    max_vel = np.array([max_v, max_w], dtype=np.float32)

    max_acc = np.array([max_a, max_aw], dtype=np.float32) * float(dt)
    if np.linalg.norm(np.abs(max_acc)) < 1e-5:
        max_acc = np.ones(2, dtype=np.float32) * 1e-5

    out = restrict_vel_acc(target, prev, max_vel, max_acc)
    return float(out[0]), float(out[1])
