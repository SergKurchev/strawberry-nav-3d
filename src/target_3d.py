import json
from pathlib import Path

import numpy as np


STOP_DISTANCE_M = 0.20


def approx_K(width: int, height: int) -> np.ndarray:
    """
    ВНИМАНИЕ: это приблизительная K для тестов, НЕ калибровка.
    Потом заменим на реальные intrinsics камеры.
    """
    fx = fy = float(max(width, height))  # грубая оценка
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


def main() -> None:
    json_path = Path("results/strawberries_sample.json")
    depth_path = Path("results/strawberries_sample_depth.npy")
    masks_path = Path("results/strawberries_sample_masks_combined.npy")

    data = json.loads(json_path.read_text())
    w = int(data["image_size"]["width"])
    h = int(data["image_size"]["height"])
    K = approx_K(w, h)

    closest_id = int(data["statistics"]["closest_strawberry_id"])

    depth = np.load(depth_path)  # meters, shape (H, W)
    masks = np.load(masks_path)  # int ids, shape (H, W)

    mask = (masks == closest_id)
    if int(mask.sum()) == 0:
        raise RuntimeError(f"mask for id={closest_id} is empty")

    z_med = float(np.median(depth[mask]))
    z_center = float(data["detections"][closest_id]["depth"]["center_meters"]) if closest_id < len(data["detections"]) else None

    # пиксель центра bbox (из json)
    det = next(d for d in data["detections"] if int(d["id"]) == closest_id)
    u = float(det["bbox"]["center_x"])
    v = float(det["bbox"]["center_y"])

    p_c = unproject(u, v, z_med, K)  # (x,y,z) в метрах в СК камеры
    dist = float(np.linalg.norm(p_c))
    remaining = max(0.0, dist - STOP_DISTANCE_M)

    print(f"closest_id={closest_id}")
    print(f"z_median_m={z_med:.3f} (z_center_json={z_center})")
    print(f"camera_point_m = [{p_c[0]:.3f}, {p_c[1]:.3f}, {p_c[2]:.3f}]")
    print(f"euclid_dist_m={dist:.3f}")
    print(f"remaining_to_{STOP_DISTANCE_M:.2f}m={remaining:.3f}")

    print("\nNOTE: K here is APPROXIMATE. Replace with real camera intrinsics later.")


if __name__ == "__main__":
    main()
