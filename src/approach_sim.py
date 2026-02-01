import numpy as np

STOP_M = 0.20
SLOW_M = 0.60
V_MAX = 0.40      # m/s
A_MAX = 0.60      # m/s^2
DT = 0.05         # s
MAX_STEPS = 500


def v_desired(dist: float) -> float:
    if dist <= STOP_M:
        return 0.0
    if dist >= SLOW_M:
        return V_MAX
    # linear ramp from 0 at STOP_M to V_MAX at SLOW_M
    return V_MAX * (dist - STOP_M) / (SLOW_M - STOP_M)


def step_speed(v: float, v_des: float) -> float:
    dv_max = A_MAX * DT
    if v_des > v:
        return min(v + dv_max, v_des)
    else:
        return max(v - dv_max, v_des)


def main() -> None:
    # цель (пример) — подставь сюда из target_3d.py
    goal = np.array([-0.108, 0.173, 0.911], dtype=np.float32)

    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # старт робота в СК камеры (условно)
    v = 0.0

    for k in range(MAX_STEPS):
        d_vec = goal - x
        dist = float(np.linalg.norm(d_vec))
        rem = max(0.0, dist - STOP_M)

        v_des = v_desired(dist)
        v = step_speed(v, v_des)

        if dist > 1e-9:
            direction = d_vec / dist
        else:
            direction = np.zeros(3, dtype=np.float32)

        x = x + direction * (v * DT)

        if k % 10 == 0 or rem <= 1e-6:
            print(f"t={k*DT:5.2f}s  dist={dist:6.3f}  v={v:5.3f}  remaining={rem:6.3f}")

        if dist <= STOP_M + 1e-6:
            print("STOP reached (<= 0.20m)")
            break


if __name__ == "__main__":
    main()
