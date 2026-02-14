from __future__ import annotations

from dataclasses import dataclass

from .config import Sim2DConfig
from .motor_math import limit_vw


@dataclass
class MotorOutput:
    v: float
    w: float


class MotorBackend:
    """Abstract motor backend. Later can be replaced with ROS-based backend."""

    def set_cmd(self, v_cmd: float, w_cmd: float, dt: float) -> MotorOutput:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state (e.g., previous velocities)."""
        return None


@dataclass
class SimMotorBackend(MotorBackend):
    cfg: Sim2DConfig
    v_prev: float = 0.0
    w_prev: float = 0.0

    def set_cmd(self, v_cmd: float, w_cmd: float, dt: float) -> MotorOutput:
        v, w = limit_vw(v_cmd, w_cmd, self.v_prev, self.w_prev, dt=dt, cfg=self.cfg)
        self.v_prev, self.w_prev = v, w
        return MotorOutput(v=v, w=w)

    def reset(self) -> None:
        self.v_prev = 0.0
        self.w_prev = 0.0


@dataclass
class RawMotorBackend(MotorBackend):
    """No motor limiting; passes commands through (debug mode)."""

    def set_cmd(self, v_cmd: float, w_cmd: float, dt: float) -> MotorOutput:
        return MotorOutput(v=float(v_cmd), w=float(w_cmd))


@dataclass
class RosStubMotorBackend(SimMotorBackend):
    """Placeholder for future ROS integration. For now uses same math limiting."""
    pass


def make_motor_backend(cfg: Sim2DConfig) -> MotorBackend:
    name = str(getattr(cfg, "MOTOR_BACKEND", "math")).lower().strip()

    if name in ("math", "sim", "default"):
        return SimMotorBackend(cfg)

    if name in ("raw", "passthrough"):
        return RawMotorBackend()

    if name in ("ros", "ros-stub", "rosstub"):
        return RosStubMotorBackend(cfg)

    # fallback
    return SimMotorBackend(cfg)
