
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src" / "risk"))

from partition import partition_mask_box, replication_seed

# Read ONLY the frozen calibration list
calib_file = project / "splits" / "calib.txt"

with open(calib_file, "r", encoding="utf-8") as f:
    calibration_pool = [
        line.strip()
        for line in f
        if line.strip()
    ]

assert len(calibration_pool) == 1186

print("REAL CALIBRATION PARTITION TEST")
print("=" * 50)
print("Calibration images:", len(calibration_pool))

# Example: reveal masks for 100 images
n = 100

for replication_id in range(3):
    seed = replication_seed(replication_id)

    mask_set, box_only_set = partition_mask_box(
        calibration_pool, n=n, seed=seed
    )

    assert len(mask_set) == n
    assert len(box_only_set) == len(calibration_pool) - n

    assert set(mask_set).isdisjoint(box_only_set)

    assert (
        set(mask_set) | set(box_only_set)
        == set(calibration_pool)
    )

    print(
        f"Replication {replication_id + 1}: "
        f"seed={seed}, "
        f"mask={len(mask_set)}, "
        f"box-only={len(box_only_set)} "
        "PASS"
    )

print("\nALL REAL CALIBRATION PARTITION TESTS PASSED!")
