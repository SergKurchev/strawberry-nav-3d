import numpy as np

from src.config import (
    STOP_M, SLOW_M, V_MAX, A_MAX, DT, MAX_STEPS,
    STOP_TOL_M, SWITCH_MIN_R, SWITCH_ADD_R,
)


def v_desired(dist_to_final: float) -> float:
    """
    dist_to_final: расстояние до конечной цели (в метрах).
    Скорость = V_MAX далеко, линейно падает до 0 при dist=STOP_M.
    """
    if dist_to_final <= STOP_M:
        return 0.0
    if dist_to_final >= SLOW_M:
        return V_MAX
    # линейный профиль на участке (STOP_M, SLOW_M)
    alpha = (dist_to_final - STOP_M) / max(1e-9, (SLOW_M - STOP_M))
    return float(np.clip(alpha * V_MAX, 0.0, V_MAX))


def step_speed(v: float, v_des: float) -> float:
    dv = v_des - v
    max_dv = A_MAX * DT
    dv = float(np.clip(dv, -max_dv, max_dv))
    return v + dv


def follow_waypoints(waypoints: list[np.ndarray]) -> None:
    """
    Простая кинематическая симуляция: движемся по 3D вейпоинтам.
    Печатает прогресс и останавливается при достижении STOP_M.
    """
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    v = 0.0
    wi = 0

    for k in range(MAX_STEPS):
        if wi >= len(waypoints):
            print("DONE (waypoints finished)")
            return

        final_vec = waypoints[-1] - x
        final_dist = float(np.linalg.norm(final_vec))
        remaining = max(0.0, final_dist - STOP_M)

        if remaining <= STOP_TOL_M:
            print("STOP reached (<= 0.20m to final)")
            return

        target = waypoints[wi]
        d_vec = target - x
        dist = float(np.linalg.norm(d_vec))

        switch_r = max(SWITCH_MIN_R, 0.5 * v * DT + SWITCH_ADD_R)
        if dist < switch_r and wi < len(waypoints) - 1:
            wi += 1
            continue

        v_des = v_desired(final_dist)
        v = step_speed(v, v_des)

        direction = d_vec / dist if dist > 1e-9 else np.zeros(3, dtype=np.float32)
        x = x + direction * (v * DT)

        if k % 10 == 0:
            print(f"t={k*DT:5.2f}s  wp={wi:02d}/{len(waypoints)-1:02d}  v={v:5.3f}  remaining={remaining:7.4f}")

    print("STOP: max steps reached")
