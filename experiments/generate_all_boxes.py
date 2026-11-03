
import csv
import json
import sys
from pathlib import Path
from collections import Counter

# Project paths
project = Path(__file__).resolve().parents[1]
data_dir = project.parent / "StructGround-UQ-data"

sys.path.insert(
    0, str(project / "src" / "segmentation")
)

from union_mask import create_class_union_mask
from bounding_boxes import analyze_mask

# Input and output paths
input_file = data_dir / "samples.json"
output_file = data_dir / "dacl10k_boxes.csv"

assert input_file.exists(), f"Missing: {input_file}"

print("Loading dacl10k annotations...")

with open(input_file, "r", encoding="utf-8") as f:
    samples = json.load(f)["samples"]

print("Total image records:", len(samples))

columns = [
    "image_name", "split", "class_name",
    "width", "height", "components",
    "damaged_pixels",
    "x_min", "y_min", "x_max", "y_max"
]

counts = Counter()
empty_masks = 0
skipped_no_labels = 0

# Process one image at a time; do not keep masks
# in memory or save them to disk.
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for index, sample in enumerate(samples, start=1):
        split = sample.get("tags", ["unknown"])[0]

        # Testdev has no segmentation ground truth.
        if split not in ("train", "validation"):
            continue

        annotations = (
            sample.get("semseg_ground_truth_filled") or {}
        )
        polygons = annotations.get("polylines", [])

        classes = sorted({
            p["label"]
            for p in polygons
            if p.get("label")
        })

        if not classes:
            skipped_no_labels += 1
            continue

        for class_name in classes:
            mask = create_class_union_mask(
                sample, class_name
            )

            result = analyze_mask(mask)
            box = result["bounding_box"]

            if box is None:
                empty_masks += 1
                continue

            x_min, y_min, x_max, y_max = box

            height, width = mask.shape

            assert 0 <= x_min < x_max <= width
            assert 0 <= y_min < y_max <= height
            assert result["damaged_pixels"] > 0

            # Verify box encloses every positive pixel.
            assert mask[:y_min, :].sum() == 0
            assert mask[y_max:, :].sum() == 0
            assert mask[:, :x_min].sum() == 0
            assert mask[:, x_max:].sum() == 0

            writer.writerow({
                "image_name": sample["name"],
                "split": split,
                "class_name": class_name,
                "width": width,
                "height": height,
                "components": result["component_count"],
                "damaged_pixels": result["damaged_pixels"],
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            })

            counts[split] += 1

        if index % 500 == 0:
            print(f"Processed {index}/{len(samples)} images")

print("\nPREPROCESSING COMPLETED")
print("=" * 45)
print("Boxes saved to:", output_file)
print("Training image-class records:", counts["train"])
print("Validation image-class records:", counts["validation"])
print("Images with no class polygons:", skipped_no_labels)
print("Class masks with no positive pixels:", empty_masks)
print("Total box records:", sum(counts.values()))
