import numpy as np


def approx_K(width: int, height: int) -> np.ndarray:
    """
    Грубая K только для тестов. Потом заменим на реальные intrinsics.
    """
    fx = fy = float(max(width, height))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def unproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def goal_point_from_results(data: dict, depth: np.ndarray, masks: np.ndarray, K: np.ndarray):
    """
    Возвращает:
      goal_p (3D в СК камеры), closest_id, z_median
    """
    closest_id = int(data["statistics"]["closest_strawberry_id"])

    det = next(d for d in data["detections"] if int(d["id"]) == closest_id)
    u = float(det["bbox"]["center_x"])
    v = float(det["bbox"]["center_y"])

    m = (masks == closest_id)
    if int(m.sum()) == 0:
        raise RuntimeError(f"empty mask for id={closest_id}")

    z_med = float(np.median(depth[m]))
    goal_p = unproject(u, v, z_med, K)
    return goal_p, closest_id, z_med
