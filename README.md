[![CI](https://github.com/pavelkuz001/strawberry-nav-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/pavelkuz001/strawberry-nav-3d/actions/workflows/ci.yml)

# strawberry-nav-3d

Pipeline: strawberry detection + depth -> 3D voxel map -> 3D A* path -> simple speed controller.

## Quick start (CPU)

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -r third_party/strawberry_detector/requirements.txt
pip install -e third_party/strawberry_detector


# run detector on sample image (first run will download models to user cache)
mkdir -p results
python -m strawberry_detector \
  --image third_party/strawberry_detector/test_images/strawberries_sample.jpg \
  --output results --save-full --device cpu

# run 3D A* pipeline on produced results
python -m src.main

# run loop-based simulation (optional)
python -m src.loop


```
