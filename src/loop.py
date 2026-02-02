from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from strawberry_detector import StrawberryDetector

from src.core.vision import approx_K, goal_point_from_results
from src.core.map3d import build_voxel_map, idx_to_point
from src.core.astar3d import astar
from src.core.controller import v_desired, step_speed
from src import config


def load_detector_result(json_path: str):
    data = json.loads(Path(json_path).read_text())
    out = data.get("output_files", {})
    depth = np.load(out["depth_map"])
    masks = np.load(out["masks_combined"])
    return data, depth, masks


def remaining_to_stop(p_robot: np.ndarray, p_goal: np.ndarray, stop_r: float) -> float:
    d = float(np.linalg.norm(p_goal - p_robot))
    return max(0.0, d - stop_r)


def resample_path_points(vmap, path_idxs):
    # Индексы -> точки (в той же СК, где vmap.origin / goal_p / start_p)
    pts = [idx_to_point(idx, vmap.origin) for idx in path_idxs]
    return [np.asarray(p, dtype=float) for p in pts]


def follow_with_feedback(waypoints, goal_p, stop_r, dt=0.1, max_t=20.0):
    """
    Простейшая симуляция: робот движется по прямой к текущему waypoint.
    Скорость задаём smooth_speed(dist_to_goal, v_prev, dt) (как у тебя в проекте).
    """
    p = np.array(waypoints[0], dtype=float)  # стартовая позиция (первая точка пути)
    wp_i = 0
    v_prev = 0.0
    t = 0.0

    def dist(a, b): return float(np.linalg.norm(a - b))

    while t <= max_t:
        rem = remaining_to_stop(p, goal_p, stop_r)
        if rem <= 1e-3:
            print(f"STOP reached (<= {stop_r:.2f}m to goal)")
            break

        # переключаем waypoint когда близко
        if wp_i < len(waypoints) - 1 and dist(p, waypoints[wp_i]) < 0.05:
            wp_i += 1

        target = waypoints[wp_i]
        d_to_target = dist(p, target)
        if d_to_target < 1e-9:
            t += dt
            continue

        # скорость: чем ближе к финалу (goal_p), тем меньше скорость
        d_to_goal = dist(p, goal_p)
        v_des = float(v_desired(d_to_goal))
        v = float(step_speed(v_prev, v_des))
        v = max(0.0, min(v, config.V_MAX))
        v_prev = v

        step = min(v * dt, d_to_target)
        direction = (target - p) / d_to_target
        p = p + direction * step

        if int((t + 1e-9) * 10) % 10 == 0:  # печать раз в 1s
            print(f"t={t:5.2f}s  wp={wp_i:02d}/{len(waypoints)-1:02d}  v={v:0.3f}  remaining={rem:0.4f}")

        t += dt

    return p


def main():
    # 1) Берём готовые результаты детектора (как сейчас) — позже заменим на live-детект
    data, depth, masks = load_detector_result("results/strawberries_sample.json")

    # 2) Цель в 3D (camera frame)
    K = approx_K(data["image_size"]["width"], data["image_size"]["height"])
    goal_p, goal_id, z_med = goal_point_from_results(data, depth, masks, K)
    closest_id = data["statistics"]["closest_strawberry_id"]
    print(f"closest_id={closest_id}")
    print(f"goal(camera frame)={goal_p}")

    # 3) Старт считаем в [0,0,0] (camera frame). Это ок для симуляции.
    start_p = np.zeros(3, dtype=float)

    # 4) Строим voxel map и планируем A*
    vmap = build_voxel_map(depth, K, start_p, goal_p)

    path_idxs = astar(vmap.start_idx, vmap.goal_idx, vmap.occ, vmap.bounds_max)
    if path_idxs is None or len(path_idxs) < 2:
        raise SystemExit("A* failed: no path")

    waypoints = resample_path_points(vmap, path_idxs)
    print(f"Waypoints: {len(waypoints)}")
    print("Final waypoint:", waypoints[-1])

    # 5) Движение с обратной связью remaining_to_stop
    stop_r = getattr(config, "STOP_RADIUS_M", None) or 0.20
    follow_with_feedback(waypoints, goal_p, stop_r=stop_r, dt=0.1, max_t=30.0)


if __name__ == "__main__":
    main()
