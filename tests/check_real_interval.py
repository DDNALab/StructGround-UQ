
import json
import sys
from pathlib import Path

import numpy as np

# Project paths
project = Path(__file__).resolve().parents[1]
data_dir = project.parent / "StructGround-UQ-data"

sys.path.insert(0, str(project / "src" / "segmentation"))
sys.path.insert(0, str(project / "src" / "risk"))

from union_mask import create_class_union_mask
from bounding_boxes import create_tight_box
from interval import compute_loss_interval

# Load real dacl10k annotation
sample_path = data_dir / "dacl10k_sample_1394.json"

with open(sample_path, "r", encoding="utf-8") as f:
    sample = json.load(f)

print("REAL DACL10K INTERVAL TEST")
print("Image:", sample["name"])
print("=" * 60)

# Test each annotated damage class
polygons = sample["semseg_ground_truth_filled"]["polylines"]
classes = sorted({p["label"] for p in polygons})

for class_name in classes:

    # Y: Real damage mask from dacl10k annotations
    Y = create_class_union_mask(sample, class_name)

    # B: Tight box enclosing the real mask
    box = create_tight_box(Y)

    assert box is not None
    assert np.count_nonzero(Y) > 0

    x_min, y_min, x_max, y_max = box

    # Confirm the real mask is inside its box
    box_mask = np.zeros_like(Y, dtype=bool)
    box_mask[y_min:y_max, x_min:x_max] = True

    assert not np.any((Y != 0) & ~box_mask)

    # Three illustrative predictions, not model outputs
    predictions = {}

    # Case 1: Prediction covers the entire box
    U1 = box_mask.astype(np.uint8)
    predictions["Full coverage"] = U1

    # Case 2: Prediction misses the entire box
    U2 = np.zeros_like(Y, dtype=np.uint8)
    predictions["No coverage"] = U2

    # Case 3: Prediction covers part of the box
    U3 = np.zeros_like(Y, dtype=np.uint8)
    U3[y_min:y_max, x_min:x_max] = (
        box_mask[y_min:y_max, x_min:x_max]
        & (np.indices(
            (y_max - y_min, x_max - x_min)
        )[1] < (x_max - x_min) // 2)
    ).astype(np.uint8)

    predictions["Partial coverage"] = U3

    print("\nClass:", class_name)
    print("Box:", box)

    for prediction_name, U in predictions.items():

        result = compute_loss_interval(U, box)

        # True binary missed-damage loss
        true_loss = int(
            np.any((Y != 0) & (U == 0))
        )

        lower = result["lower_loss"]
        upper = result["upper_loss"]

        assert lower <= true_loss <= upper, (
            f"Interval failed for {class_name}, "
            f"{prediction_name}"
        )

        print(
            f"  {prediction_name:18} "
            f"interval=[{lower},{upper}] "
            f"true_loss={true_loss} "
            f"PASS"
        )

print("\nALL REAL DACL10K INTERVAL TESTS PASSED!")
