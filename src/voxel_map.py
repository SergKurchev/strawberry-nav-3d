import numpy as np
import json
from pathlib import Path


def approx_K(width: int, height: int) -> np.ndarray:
    fx = fy = float(max(width, height))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def depth_to_points(depth: np.ndarray, K: np.ndarray, stride: int = 4, z_max: float = 2.0) -> np.ndarray:
    """
    depth: (H,W) meters
    return: (N,3) points in camera frame
    """
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    us = np.arange(0, W, stride, dtype=np.int32)
    vs = np.arange(0, H, stride, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)

    z = depth[vv, uu].astype(np.float32)
    valid = np.isfinite(z) & (z > 0.0) & (z < z_max)

    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    z = z[valid]

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    pts = np.stack([x, y, z], axis=1)
    return pts


def points_to_voxels(pts: np.ndarray, voxel: float = 0.05):
    """
    Возвращает:
      occ: set of (ix,iy,iz)
      origin: (minx,miny,minz)
    """
    if pts.shape[0] == 0:
        return set(), np.zeros(3, dtype=np.float32)

    mins = pts.min(axis=0)
    idx = np.floor((pts - mins) / voxel).astype(np.int32)

    occ = set(map(tuple, idx))
    return occ, mins


def main():
    data = json.loads(Path("results/strawberries_sample.json").read_text())
    w = int(data["image_size"]["width"])
    h = int(data["image_size"]["height"])
    K = approx_K(w, h)

    depth = np.load("results/strawberries_sample_depth.npy")
    pts = depth_to_points(depth, K, stride=4, z_max=2.0)

    occ, origin = points_to_voxels(pts, voxel=0.05)

    print("depth shape:", depth.shape)
    print("points:", pts.shape[0])
    print("occupied voxels:", len(occ))
    print("origin (min x,y,z):", origin)


if __name__ == "__main__":
    main()
