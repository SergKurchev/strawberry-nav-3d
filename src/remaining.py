import json
from pathlib import Path


STOP_DISTANCE_CM = 20.0


def main() -> None:
    p = Path("results/strawberries_sample.json")
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}. Run detector first.")

    data = json.loads(p.read_text())

    closest_cm = float(data["statistics"]["closest_distance_cm"])
    closest_id = int(data["statistics"]["closest_strawberry_id"])
    remaining_cm = max(0.0, closest_cm - STOP_DISTANCE_CM)

    print(f"closest_id={closest_id}")
    print(f"closest_cm={closest_cm:.2f}")
    print(f"remaining_to_{STOP_DISTANCE_CM:.0f}cm={remaining_cm:.2f}")


if __name__ == "__main__":
    main()
