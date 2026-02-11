import numpy as np


def cam_xyz_to_robot_xy(xyz_cam: np.ndarray) -> np.ndarray:
    """
    Camera frame (Depth-style):
      X_cam: right
      Y_cam: down
      Z_cam: forward

    Robot local 2D frame (top-down):
      x: forward
      y: left

    Default mapping:
      x = Z_cam
      y = -X_cam
    """
    X = float(xyz_cam[0])
    Z = float(xyz_cam[2])
    return np.array([Z, -X], dtype=np.float32)


def robot_xy_to_cam_xyz(p_robot: np.ndarray) -> np.ndarray:
    """
    Inverse mapping for 2D sim (assume Y_cam=0):
      X_cam = -y_left
      Z_cam =  x_fwd
      Y_cam = 0
    """
    x_fwd = float(p_robot[0])
    y_left = float(p_robot[1])
    return np.array([-y_left, 0.0, x_fwd], dtype=np.float32)


def robot_xy_to_world_xy(p_robot: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    """
    pose_world = [x, y, theta], theta radians.
    """
    xr, yr = float(p_robot[0]), float(p_robot[1])
    x0, y0, th = float(pose_world[0]), float(pose_world[1]), float(pose_world[2])

    c = np.cos(th)
    s = np.sin(th)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    return np.array([x0, y0], dtype=np.float32) + R @ np.array([xr, yr], dtype=np.float32)


def world_xy_to_robot_xy(p_world: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    """
    Inverse transform: world -> robot local.
    """
    xw, yw = float(p_world[0]), float(p_world[1])
    x0, y0, th = float(pose_world[0]), float(pose_world[1]), float(pose_world[2])

    dx = xw - x0
    dy = yw - y0

    c = np.cos(th)
    s = np.sin(th)
    # R(-th) = R(th)^T
    R_T = np.array([[c,  s],
                    [-s, c]], dtype=np.float32)

    return R_T @ np.array([dx, dy], dtype=np.float32)
