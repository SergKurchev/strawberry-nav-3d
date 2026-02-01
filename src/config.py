# --- Map / A* ---
VOXEL = 0.05      # meters per voxel
STRIDE = 4        # depth sampling stride (pixels)
Z_MAX = 2.0       # max depth to consider (meters)
PAD_VOX = 5       # padding around bounds (voxels)

# --- Approach / smoothing ---
STOP_M = 0.20     # stop distance to strawberry (meters)
SLOW_M = 0.60     # begin slowing down when closer than this (meters)
V_MAX = 0.40      # max speed (m/s)
A_MAX = 0.60      # max accel/decel (m/s^2)
DT = 0.05         # control timestep (s)
MAX_STEPS = 800   # sim safety cap

# tolerances
STOP_TOL_M = 0.005   # 5mm tolerance for stopping
SWITCH_MIN_R = 0.06  # min waypoint switch radius (m)
SWITCH_ADD_R = 0.04  # + radius part (m)
