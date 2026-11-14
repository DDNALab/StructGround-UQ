"""Week 3 Step 10: classwise, thresholdwise pooled IoU on frozen splits.

Do not inspect or use results/iou_test.csv for model selection.
Images with zero annotation polylines are excluded as annotation-ambiguous.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "StructGround-UQ-data"
sys.path.insert(0, str(PROJECT / "src" / "segmentation"))
from union_mask import create_class_union_mask

# This is the publisher's output channel order, NOT project alphabetical IDs.
CHECKPOINT_CLASSES = (
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust",
    "Rockpocket", "Hollowareas", "Cavity", "Spalling",
    "Graffiti", "Weathering", "Restformwork", "ExposedRebars",
    "Bearing", "EJoint", "Drainage", "PEquipment", "JTape",
    "WConccor",
)


def read_thresholds():
    path = PROJECT / "configs" / "thresholds.yaml"
    content = path.read_text(encoding="utf-8")
    values = [float(match) for match in re.findall(r"^\s*-\s*(0?\.\d+)\s*$", content, flags=re.MULTILINE)]
    if len(values) != 25 or values != sorted(set(values)):
        raise ValueError("Frozen threshold grid must have 25 unique sorted values")
    if not np.allclose(values, np.arange(2, 99, 4) / 100):
        raise ValueError("Frozen thresholds differ from the Step 7 grid")
    return np.asarray(values, dtype=np.float64)


def read_split(split):
    path = PROJECT / "splits" / f"{split}.txt"
    entries = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    expected_count = 1186 if split == "calib" else 1187
    if len(entries) != expected_count or len(set(entries)) != len(entries):
        raise ValueError(f"Frozen {split} list has unexpected count or duplicates")
    return [Path(x.replace('\\', '/')).stem for x in entries]


def read_annotations():
    path = DATA / "samples.json"
    records = json.loads(path.read_text(encoding="utf-8"))["samples"]
    by_stem = {}
    for sample in records:
        stem = Path(sample["filepath"].replace("\\", "/")).stem
        if stem in by_stem:
            raise ValueError(f"Duplicate annotation record: {stem}")
        by_stem[stem] = sample
    return by_stem


def validate_score_manifest(split, stems):
    folder = PROJECT / "results" / "scores" / split
    manifest_path = folder / "manifest_sha256.txt"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    expected = {stem + ".npy" for stem in stems}
    listed = [line.split("  ", 1)[1] for line in lines]
    actual = {p.name for p in folder.glob("*.npy")}
    if len(lines) != len(stems) or len(listed) != len(set(listed)) or set(listed) != expected or actual != expected:
        raise ValueError(f"{split} score manifest/files do not exactly match frozen split")
    return folder


def histogram_counts(scores, mask, thresholds):
    """Return counts by number of thresholds each pixel meets (0..25).

    If a pixel meets threshold[j], its bin index is > j.
    This permits exact pooled IoU for every threshold in one pass.
    """
    bins = np.searchsorted(thresholds, np.asarray(scores).ravel(), side="right")
    if not np.isfinite(scores).all() or np.min(scores) < 0 or np.max(scores) > 1:
        raise ValueError("Invalid score values")
    mask = np.asarray(mask, dtype=bool)
    return (
        np.bincount(bins, minlength=len(thresholds) + 1),
        np.bincount(bins[mask.ravel()], minlength=len(thresholds) + 1),
    )


def compute(split):
    thresholds = read_thresholds()
    stems = read_split(split)
    annotations = read_annotations()
    folder = validate_score_manifest(split, stems)

    n_classes, n_bins = len(CHECKPOINT_CLASSES), len(thresholds) + 1
    all_hist = np.zeros((n_classes, n_bins), dtype=np.int64)
    gt_hist = np.zeros((n_classes, n_bins), dtype=np.int64)

    skipped = 0
    processed = 0

    for i, stem in enumerate(stems, start=1):
        sample = annotations.get(stem)
        if sample is None:
            raise ValueError(f"Missing annotation record: {stem}")
        annotation = sample.get("semseg_ground_truth_filled")
        if annotation is None or not annotation.get("polylines"):
            skipped += 1
            continue

        height = int(sample["metadata"]["height"])
        width = int(sample["metadata"]["width"])
        scores = np.load(folder / f"{stem}.npy", mmap_mode="r", allow_pickle=False)
        if scores.shape != (n_classes, height, width):
            raise ValueError(f"Score/ground-truth dimension mismatch: {stem}: {scores.shape} vs {(n_classes, height, width)}")

        for channel, class_name in enumerate(CHECKPOINT_CLASSES):
            mask = create_class_union_mask(sample, class_name)
            counts, true_counts = histogram_counts(scores[channel], mask, thresholds)
            all_hist[channel] += counts
            gt_hist[channel] += true_counts

        processed += 1
        del scores
        if i % 50 == 0 or i == len(stems):
            print(f"{split}: checked {i}/{len(stems)} images", flush=True)

    if not processed:
        raise ValueError("No images with usable polygon annotations")

    output = PROJECT / "results" / f"iou_{split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("class", "threshold", "IoU"))
        for channel, class_name in enumerate(CHECKPOINT_CLASSES):
            gt_count = int(gt_hist[channel].sum())
            # Number of score pixels >= each threshold.
            pred_count = all_hist[channel].sum() - np.cumsum(all_hist[channel][:-1])
            true_pos = gt_count - np.cumsum(gt_hist[channel][:-1])
            union = gt_count + pred_count - true_pos
            for j, threshold in enumerate(thresholds):
                # Undefined when both pooled masks are empty.
                iou = "" if union[j] == 0 else format(float(true_pos[j] / union[j]), ".12g")
                writer.writerow((class_name, f"{threshold:.2f}", iou))

    with output.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = {(r["class"], r["threshold"]) for r in rows}
    expected_pairs = {(name, f"{t:.2f}") for name in CHECKPOINT_CLASSES for t in thresholds}
    if len(rows) != 475 or pairs != expected_pairs:
        raise ValueError("IoU CSV lacks exactly one row per class/threshold")

    print(f"{split}: images with annotations used: {processed}")
    print(f"{split}: images with no polygons excluded: {skipped}")
    print(f"{split}: CSV written: {output}")
    print(f"{split}: class-threshold rows verified: {len(rows)}")
    if split == "test":
        print("TEST IoU VALUES NOT DISPLAYED. Do not inspect or use this CSV for decisions.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("calib", "test"), required=True)
    args = parser.parse_args()
    compute(args.split)


if __name__ == "__main__":
    main()
