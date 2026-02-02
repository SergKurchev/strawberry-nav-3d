[![CI](https://github.com/pavelkuz001/strawberry-nav-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/pavelkuz001/strawberry-nav-3d/actions/workflows/ci.yml)

# strawberry-nav-3d

Prototype pipeline (per tech spec): **strawberry detection + depth → 3D voxel map → 3D A\* path → waypoints → simple speed controller**.

Key idea: the repo stays clean after `git clone`. Large assets (weights, Depth-Anything-V2 repo) are stored in a **per-user cache** (cross-platform), not committed to git.

---

## What is implemented (per Tech Spec)

### ✅ 1) “git clone → run”
- Repository can be cloned normally (`git clone`).
- Heavy assets are not tracked by git (ignored) and are downloaded on demand.

### ✅ 2) Strawberry detector + depth estimation
- CLI tool: `python -m strawberry_detector`
- Outputs: `*.json`, `*_depth.npy`, `*_masks_combined.npy`, individual masks, visualization image.
- First run downloads:
  - Depth-Anything-V2 repository
  - Depth weights
  - YOLO weights

### ✅ 3) 3D planning & approach
- Build voxel occupancy from depth.
- Run **3D A\*** on voxel grid.
- Convert path to **waypoints** in camera frame.
- Follow waypoints with a simple speed controller (slows down near goal).
- Two entrypoints:
  - `python -m src.main` — main pipeline run (reads detector outputs from `results/`)
  - `python -m src.loop` — loop-based simulation of waypoint following (optional)

### ✅ 4) Cross-platform caching of models
- Uses `platformdirs` to store large files in user cache directory.
- No model weights are committed to git.

### ✅ 5) CI smoke test (Ubuntu)
- GitHub Actions checks: install deps, compile, import smoke.

---

## Repo structure

- `src/`
  - `main.py` — end-to-end demo: load results → goal point → voxel map → A* → follow
  - `loop.py` — loop simulation runner
  - `core/vision.py` — goal point from results (camera frame), approximate intrinsics (for demo)
  - `core/map3d.py` — voxel map build + index conversions
  - `core/astar3d.py` — 3D A* implementation
  - `core/controller.py` — simple speed control + waypoint following
- `third_party/strawberry_detector/`
  - packaged detector module (installed editable)
- `results/`
  - local outputs (ignored by git)

---

## Quick start (CPU)

### 0) Prerequisites
- Python 3.9+ (3.10/3.11 also OK)
- `git`
- Internet access for first model download

### 1) Clone + venv
```bash
git clone https://github.com/pavelkuz001/strawberry-nav-3d.git
cd strawberry-nav-3d

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell

