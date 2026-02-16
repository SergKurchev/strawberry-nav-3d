#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

python -m strawberry_detector \
  --image third_party/strawberry_detector/test_images/strawberries_sample.jpg \
  --output results --save-full --device cpu

python -m src.sim2d.run \
  --json results/strawberries_sample.json \
  --motor-backend runbot \
  --init-theta 3.1416 \
  --save-fig results/sim2d_tz.png

echo "OK: saved results/sim2d_tz.png"
