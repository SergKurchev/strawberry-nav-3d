import numpy as np


def pixel_depth_to_cam_xyz(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """
    Convert pixel coordinates (u,v) + depth (meters) into camera-frame metric coordinates (X,Y,Z).
    K is 3x3 intrinsics matrix:
        [fx  0 cx]
        [ 0 fy cy]
        [ 0  0  1]
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    Z = float(depth_m)
    X = (float(u) - cx) * Z / fx
    Y = (float(v) - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)
