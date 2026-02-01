import json
from pathlib import Path
import heapq
import numpy as np


VOXEL = 0.05      # meters
STRIDE = 4        # sampling step in depth image
Z_MAX = 2.0       # meters
PAD_VOX = 5       # padding voxels around bounds


def approx_K(width: int, height: int) -> np.ndarray:
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


def depth_to_points(depth: np.ndarray, K: np.ndarray, stride: int = 4, z_max: float = 2.0) -> np.ndarray:
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


def to_idx(p: np.ndarray, origin: np.ndarray) -> tuple[int, int, int]:
    ijk = np.floor((p - origin) / VOXEL).astype(np.int32)
    return int(ijk[0]), int(ijk[1]), int(ijk[2])


def build_occupancy(pts: np.ndarray, origin: np.ndarray) -> set[tuple[int, int, int]]:
    ijk = np.floor((pts - origin) / VOXEL).astype(np.int32)
    return set(map(tuple, ijk))


def neighbors26():
    res = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                cost = (dx*dx + dy*dy + dz*dz) ** 0.5
                res.append((dx, dy, dz, cost))
    return res


NEI = neighbors26()


def heuristic(a, b) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5


def astar(start, goal, occ: set, bounds_max):
    """
    bounds: 0<=x<=maxx etc (inclusive)
    """
    maxx, maxy, maxz = bounds_max

    def in_bounds(n):
        x, y, z = n
        return 0 <= x <= maxx and 0 <= y <= maxy and 0 <= z <= maxz

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    came = {start: None}
    g = {start: 0.0}

    while open_heap:
        f, gcur, cur = heapq.heappop(open_heap)
        if cur == goal:
            # reconstruct
            path = []
            n = cur
            while n is not None:
                path.append(n)
                n = came[n]
            path.reverse()
            return path

        for dx, dy, dz, step in NEI:
            nxt = (cur[0]+dx, cur[1]+dy, cur[2]+dz)
            if not in_bounds(nxt):
                continue
            if nxt in occ:
                continue
            ng = gcur + step
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng + heuristic(nxt, goal), ng, nxt))

    return None


def main():
    data = json.loads(Path("results/strawberries_sample.json").read_text())
    w = int(data["image_size"]["width"])
    h = int(data["image_size"]["height"])
    K = approx_K(w, h)

    depth = np.load("results/strawberries_sample_depth.npy")
    pts = depth_to_points(depth, K, stride=STRIDE, z_max=Z_MAX)

    # цель (closest)
    closest_id = int(data["statistics"]["closest_strawberry_id"])
    det = next(d for d in data["detections"] if int(d["id"]) == closest_id)
    u = float(det["bbox"]["center_x"])
    v = float(det["bbox"]["center_y"])
    mask_id = closest_id
    masks = np.load("results/strawberries_sample_masks_combined.npy")
    m = (masks == mask_id)
    z_med = float(np.median(depth[m]))
    goal_p = unproject(u, v, z_med, K)

    # старт (камера в (0,0,0))
    start_p = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # origin так, чтобы включить и start, и point cloud
    mins = pts.min(axis=0)
    origin = np.minimum(mins, start_p) - 2.0 * VOXEL

    occ = build_occupancy(pts, origin)

    start = to_idx(start_p, origin)
    goal = to_idx(goal_p, origin)

    # bounds: по максимумам индексов (points + goal + start) + pad
    ijk_pts = np.floor((pts - origin) / VOXEL).astype(np.int32)
    max_ijk = ijk_pts.max(axis=0)
    max_ijk = np.maximum(max_ijk, np.array(start, dtype=np.int32))
    max_ijk = np.maximum(max_ijk, np.array(goal, dtype=np.int32))
    max_ijk = max_ijk + PAD_VOX
    bounds_max = (int(max_ijk[0]), int(max_ijk[1]), int(max_ijk[2]))

    print("start idx:", start)
    print("goal  idx:", goal)
    print("occupied voxels:", len(occ))
    print("bounds_max:", bounds_max)

    path = astar(start, goal, occ, bounds_max)
    if path is None:
        print("NO PATH")
        return

    print("PATH FOUND. length (nodes):", len(path))
    print("first:", path[0], " last:", path[-1])


if __name__ == "__main__":
    main()
