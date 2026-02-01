from dataclasses import dataclass
import numpy as np

from src.config import VOXEL, STRIDE, Z_MAX, PAD_VOX


@dataclass
class VoxelMap:
    origin: np.ndarray
    occ: set
    bounds_max: tuple
    start_idx: tuple
    goal_idx: tuple


def depth_to_points(depth: np.ndarray, K: np.ndarray, stride: int = STRIDE, z_max: float = Z_MAX) -> np.ndarray:
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
    return np.stack([x, y, z], axis=1)


def point_to_idx(p: np.ndarray, origin: np.ndarray) -> tuple:
    ijk = np.floor((p - origin) / VOXEL).astype(np.int32)
    return int(ijk[0]), int(ijk[1]), int(ijk[2])


def idx_to_point(idx: tuple, origin: np.ndarray) -> np.ndarray:
    return origin + (np.array(idx, dtype=np.float32) + 0.5) * VOXEL


def build_voxel_map(depth: np.ndarray, K: np.ndarray, start_p: np.ndarray, goal_p: np.ndarray) -> VoxelMap:
    pts = depth_to_points(depth, K, stride=STRIDE, z_max=Z_MAX)
    if pts.shape[0] == 0:
        raise RuntimeError("No valid depth points to build voxel map")

    mins = pts.min(axis=0)
    origin = np.minimum(mins, start_p) - 2.0 * VOXEL

    occ = set(map(tuple, np.floor((pts - origin) / VOXEL).astype(np.int32)))

    start_idx = point_to_idx(start_p, origin)
    goal_idx = point_to_idx(goal_p, origin)

    ijk_pts = np.floor((pts - origin) / VOXEL).astype(np.int32)
    max_ijk = ijk_pts.max(axis=0)
    max_ijk = np.maximum(max_ijk, np.array(start_idx, dtype=np.int32))
    max_ijk = np.maximum(max_ijk, np.array(goal_idx, dtype=np.int32))
    max_ijk = max_ijk + PAD_VOX

    bounds_max = (int(max_ijk[0]), int(max_ijk[1]), int(max_ijk[2]))
    return VoxelMap(origin=origin, occ=occ, bounds_max=bounds_max, start_idx=start_idx, goal_idx=goal_idx)
