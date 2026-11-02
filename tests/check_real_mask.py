
import json
import sys
from pathlib import Path

import numpy as np

# Project directory
project = Path(__file__).resolve().parents[1]

# Import our union mask function
sys.path.insert(
    0, str(project / "src" / "segmentation")
)

from union_mask import create_class_union_mask

# Real annotation saved outside Git repository
sample_path = (
    project.parent
    / "StructGround-UQ-data"
    / "dacl10k_sample_1394.json"
)

# Load real dacl10k annotation
with open(sample_path, "r") as f:
    sample = json.load(f)

polygons = sample["semseg_ground_truth_filled"]["polylines"]

# Get unique class names
classes = sorted(set(p["label"] for p in polygons))

print("REAL DACL10K UNION MASK TEST")
print("=" * 55)
print("Image:", sample["name"])
print("Dimensions:",
      sample["metadata"]["width"],
      "x",
      sample["metadata"]["height"])
print("Classes:", classes)
print("Number of polygons:", len(polygons))

print("\nPER-CLASS UNION MASK RESULTS")
print("-" * 55)

for class_name in classes:

    mask = create_class_union_mask(
        sample,
        class_name
    )

    damaged_pixels = int(mask.sum())
    unique_values = np.unique(mask)

    print(f"\nClass: {class_name}")
    print("Mask shape:", mask.shape)
    print("Damaged pixels:", damaged_pixels)
    print("Unique values:", unique_values)

    # Verify the result
    assert mask.shape == (800, 1200)
    assert set(unique_values).issubset({0, 1})
    assert damaged_pixels > 0

print("\nALL REAL DATASET TESTS PASSED!")
