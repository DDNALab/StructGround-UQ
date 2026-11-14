
from pathlib import Path

import run_scoring as scorer

# Reuse the same frozen model and scoring pipeline.
# Change only the input split and output location.

scorer.EXPECTED_COUNT = 1187
scorer.CALIB_LIST = scorer.PROJECT / "splits" / "test.txt"
scorer.IMAGE_DIR = scorer.DATA / "test_images"
scorer.OUTPUT_DIR = (
    scorer.PROJECT / "results" / "scores" / "test"
)


def verify_test_inputs():
    names = scorer.load_calibration_list()

    expected = {
        name if name.lower().endswith(".jpg")
        else name + ".jpg"
        for name in names
    }

    actual = {
        path.name
        for path in scorer.IMAGE_DIR.glob("*.jpg")
    }

    if expected != actual:
        raise ValueError("Test image filenames do not match")

    if scorer.sha256(scorer.MODEL_PATH) != scorer.MODEL_HASH:
        raise ValueError("Frozen checkpoint hash mismatch")

    config = (
        scorer.PROJECT / "configs" / "scoring.yaml"
    ).read_text(encoding="utf-8")

    if "seed: 2026" not in config:
        raise ValueError("Scoring seed mismatch")

    print("TEST PREFLIGHT PASSED")
    print("Frozen test images:", len(names))
    print("Checkpoint SHA-256: MATCH")
    print("Random seed:", scorer.SEED)
    print("No test annotations or metrics accessed")

    return names


if __name__ == "__main__":
    names = verify_test_inputs()

    # Uses the same score_image(), model loader,
    # frozen weights, and SHA-256 manifest creation.
    scorer.run_scoring(names)
