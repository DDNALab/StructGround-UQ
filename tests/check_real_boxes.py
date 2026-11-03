
import json
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[1]

sys.path.insert(
    0, str(project / "src" / "segmentation")
)

from union_mask import create_class_union_mask
from bounding_boxes import analyze_mask

sample_path = (
    project.parent
    / "StructGround-UQ-data"
    / "dacl10k_sample_1394.json"
)

with open(sample_path, "r") as f:
    sample = json.load(f)

polygons = sample["semseg_ground_truth_filled"]["polylines"]
classes = sorted({p["label"] for p in polygons})

print("IMAGE:", sample["name"])
print("=" * 65)

for class_name in classes:
    mask = create_class_union_mask(sample, class_name)
    result = analyze_mask(mask)

    print(f"\nClass: {class_name}")
    print("Components:", result["component_count"])
    print("Tight box:", result["bounding_box"])
    print("Damaged pixels:", result["damaged_pixels"])

    assert result["bounding_box"] is not None
    assert result["damaged_pixels"] > 0

print("\nALL REAL BOUNDING-BOX TESTS PASSED!")
