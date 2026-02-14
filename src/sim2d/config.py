"""
2D top-down simulation config.

Goal: keep parameters in one place so later users can tweak without touching logic.
No ROS dependency here; later we can add a ROS backend that matches the same interfaces.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Sim2DConfig:
    # --- Camera intrinsics / image size (for pixel -> camera XYZ) ---
    IMAGE_W: int = 1024
    IMAGE_H: int = 1024
    FX: float = 886.81
    FY: float = 886.81
    CX: float = 512.0
    CY: float = 512.0

    # Image size (pixels)
    IMG_W: int = 1024
    IMG_H: int = 1024

    # Axis mapping from camera frame to our top-down/world frame (adjust if mirrored/flipped)
    AXIS_SWAP_XY: bool = False
    AXIS_FLIP_X: bool = False
    AXIS_FLIP_Y: bool = False
    AXIS_FLIP_Z: bool = False

    # --- Target selection / stopping / slowdown ---
    TARGET_SELECTION: str = "closest"   # "closest" for closest_strawberry_id
    STOP_RADIUS_M: float = 0.20         # stop at 20 cm from berry center
    STOP_EPS_M: float = 1e-3  # numeric tolerance for stop boundary (1mm)
    SLOWDOWN_START_M: float = 0.50      # start braking 50 cm before stop point
    DISTANCE_METRIC: str = "euclid_3d"  # "euclid_3d" or "xy_2d"

    # --- Detector update timing (simulate delays) ---
    DETECTOR_RATE_HZ: float = 5.0       # how often "camera/detector json" updates
    DETECTOR_LATENCY_S: float = 0.0     # artificial latency (seconds)

    # --- Measurement noise (set to 0 for now, but configurable) ---
    NOISE_UV_STD_PX: float = 0.0        # noise on bbox center (pixels)
    NOISE_DEPTH_STD_M: float = 0.0      # noise on depth (meters)

    # --- Simulation / control loop ---
    SIM_DT: float = 0.05                # 20 Hz simulation step
    CONTROL_RATE_HZ: float = 20.0       # control loop rate (should match SIM_DT)

    # Velocity/accel limits (match your current style)
    MAX_V: float = 0.40                 # m/s
    MAX_A: float = 0.60                 # m/s^2
    MAX_W: float = 0.8             # max angular speed (rad/s)
    MAX_W_ACC: float = 1.6         # max angular accel (rad/s^2)

    # --- Odometry / differential drive defaults (tunable) ---
    WHEEL_BASE_M: float = 0.23          # distance between wheels (m)
    WHEEL_RADIUS_M: float = 0.03        # wheel radius (m)

    # --- Integration toggles (future) ---
    MOTOR_BACKEND: str = "sim"          # "sim" now, "ros" later
    POSE_SOURCE: str = "odometry"       # pose updates from sent commands

    # Reproducibility
    RNG_SEED: int = 0


def K_from_cfg(cfg: Sim2DConfig) -> np.ndarray:
    """Build camera intrinsic matrix K."""
    return np.array(
        [
            [cfg.FX, 0.0, cfg.CX],
            [0.0, cfg.FY, cfg.CY],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
