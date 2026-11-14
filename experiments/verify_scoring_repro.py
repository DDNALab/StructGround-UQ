"""Fresh inference reproducibility check for StructGround-UQ Week 3 Step 11.

Recomputes scores into one disposable .npy file at a time, compares its
SHA-256 with the original score manifest, then deletes the temporary file.
Never reads annotations or calculates evaluation metrics.
"""
import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
import run_scoring as scorer

EXPECTED = {"calib": 1186, "test": 1187}


def get_inputs(split):
    names_path = PROJECT / "splits" / f"{split}.txt"
    names = [x.strip() for x in names_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    filenames = [n if n.lower().endswith(".jpg") else n + ".jpg" for n in names]
    if len(filenames) != EXPECTED[split] or len(set(filenames)) != len(filenames):
        raise ValueError("Unexpected count or duplicate filenames in frozen split")
    images = scorer.DATA / f"{split}_images"
    if {p.name for p in images.glob("*.jpg")} != set(filenames):
        raise ValueError("Image filenames do not exactly match frozen split")
    return images, filenames


def get_manifest(split, filenames):
    folder = PROJECT / "results" / "scores" / split
    manifest = folder / "manifest_sha256.txt"
    hashes = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in hashes or len(digest) != 64:
            raise ValueError("Duplicate name or invalid hash in original manifest")
        hashes[name] = digest
    expected = {Path(name).stem + ".npy" for name in filenames}
    if set(hashes) != expected or {p.name for p in folder.glob("*.npy")} != expected:
        raise ValueError("Original score cache or manifest does not match split")
    return hashes


def seed_everything():
    random.seed(scorer.SEED)
    np.random.seed(scorer.SEED)
    torch.manual_seed(scorer.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(scorer.SEED)


def run(split):
    # Verify the frozen configuration and checkpoint before inference.
    scorer.EXPECTED_COUNT = EXPECTED[split]
    scorer.CALIB_LIST = PROJECT / "splits" / f"{split}.txt"
    scorer.IMAGE_DIR = scorer.DATA / f"{split}_images"
    scorer.OUTPUT_DIR = PROJECT / "results" / "scores" / split
    if split == "calib":
        scorer.preflight()
    else:
        if scorer.sha256(scorer.MODEL_PATH) != scorer.MODEL_HASH:
            raise ValueError("Frozen checkpoint hash mismatch")
        scoring_config = (PROJECT / "configs" / "scoring.yaml").read_text(encoding="utf-8")
        if "seed: 2026" not in scoring_config:
            raise ValueError("Frozen scoring seed mismatch")
        print("Test preflight: checkpoint and fixed seed verified; no annotations accessed")

    images, filenames = get_inputs(split)
    original_hashes = get_manifest(split, filenames)
    seed_everything()

    print(f"Recomputing {split} scores from scratch (no cached scores reused)", flush=True)
    start = time.perf_counter()
    model, device = scorer.load_model()
    print("Inference device:", device, flush=True)

    # Temporary output outside the original score cache.
    # Only one regenerated .npy exists at a time.
    tmp_dir = PROJECT / "results" / "scores" / "repro_temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    verified = 0
    mismatches = []

    for i, filename in enumerate(filenames, 1):
        image_path = images / filename
        score = scorer.score_image(image_path, model)
        if score.ndim != 3 or score.shape[0] != 19:
            raise ValueError(f"Wrong prediction shape for {filename}")
        if not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
            raise ValueError(f"Invalid prediction values for {filename}")

        fd, temporary = tempfile.mkstemp(prefix="repro_", suffix=".npy", dir=tmp_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, score, allow_pickle=False)
            new_hash = scorer.sha256(temporary)
        finally:
            Path(temporary).unlink(missing_ok=True)

        output_name = Path(filename).stem + ".npy"
        if new_hash == original_hashes[output_name]:
            verified += 1
        else:
            mismatches.append(output_name)

        del score
        if i % 50 == 0 or i == len(filenames):
            print(f"{split}: freshly scored and hash-checked {i}/{len(filenames)}", flush=True)

    elapsed = time.perf_counter() - start
    summary = {
        "split": split,
        "seed": scorer.SEED,
        "images": len(filenames),
        "matching_hashes": verified,
        "mismatching_hashes": len(mismatches),
        "reproducible": not mismatches and verified == len(filenames),
        "rerun_elapsed_seconds": round(elapsed, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pytorch": torch.__version__,
    }
    # Print only timing and file-integrity results, never IoU or other test metrics.
    print("REPRODUCIBILITY SUMMARY")
    print(json.dumps(summary, indent=2))
    if mismatches:
        print("First mismatched filenames:", mismatches[:5])
    results_file = PROJECT / "results" / f"week3_repro_{split}.json"
    results_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Summary saved:", results_file)
    if mismatches:
        raise SystemExit("Hash mismatches found; original files were not changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("calib", "test"), required=True)
    run(parser.parse_args().split)
