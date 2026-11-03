
import json
import random
import hashlib
from pathlib import Path

# Fixed random seed
SEED = 2026

# Proposed split percentages
TRAIN_RATIO = 0.70
CALIB_RATIO = 0.15

project = Path(__file__).resolve().parents[1]
data_dir = project.parent / "StructGround-UQ-data"

input_file = data_dir / "samples.json"
split_dir = project / "splits"

if not input_file.exists():
    raise FileNotFoundError(input_file)

print("Loading dacl10k annotations...")

with open(input_file, "r", encoding="utf-8") as f:
    samples = json.load(f)["samples"]

# Use only originally labeled train/validation images.
# Original testdev is excluded because masks are unavailable.
image_names = []

for sample in samples:
    tags = sample.get("tags", [])

    if (
        "train" in tags
        or "validation" in tags
    ):
        image_names.append(sample["name"])

# Every image must appear exactly once
if len(image_names) != len(set(image_names)):
    raise ValueError("Duplicate image names detected")

image_names = sorted(image_names)

print("Labeled images:", len(image_names))

assert len(image_names) == 7910

# Source/inspection IDs were not identified in Week 1.
# Fallback: each image is its own group.
groups = [[name] for name in image_names]

# Deterministic group-level shuffle
rng = random.Random(SEED)
rng.shuffle(groups)

n = len(groups)

n_train = int(n * TRAIN_RATIO)
n_calib = int(n * CALIB_RATIO)

train_groups = groups[:n_train]
calib_groups = groups[n_train:n_train + n_calib]
test_groups = groups[n_train + n_calib:]

def flatten(group_list):
    return sorted(
        name
        for group in group_list
        for name in group
    )

train = flatten(train_groups)
calib = flatten(calib_groups)
test = flatten(test_groups)

# Check that no images overlap
assert not (set(train) & set(calib))
assert not (set(train) & set(test))
assert not (set(calib) & set(test))

assert set(train) | set(calib) | set(test) == set(image_names)

split_dir.mkdir(exist_ok=True)

def save_split(filename, names):
    path = split_dir / filename

    # Never silently overwrite frozen splits
    if path.exists():
        raise FileExistsError(
            f"Frozen split already exists: {path}"
        )

    content = "".join(name + "\n" for name in names)
    encoded = content.encode("utf-8")

    path.write_bytes(encoded)

    # Calculate hash from bytes just generated.
    # No need to reopen or inspect the test list.
    digest = hashlib.sha256(encoded).hexdigest()

    return digest

# Refuse to regenerate if any split artifact exists
files = [
    "train.txt",
    "calib.txt",
    "test.txt",
    "sha256.txt"
]

if any((split_dir / name).exists() for name in files):
    raise FileExistsError(
        "Split files already exist. "
        "Do not overwrite frozen splits."
    )

train_hash = save_split("train.txt", train)
calib_hash = save_split("calib.txt", calib)
test_hash = save_split("test.txt", test)

hash_manifest = (
    f"{train_hash}  train.txt\n"
    f"{calib_hash}  calib.txt\n"
    f"{test_hash}  test.txt\n"
)

(split_dir / "sha256.txt").write_text(
    hash_manifest,
    encoding="utf-8"
)

print("\nSPLITS CREATED SUCCESSFULLY")
print("=" * 45)
print("Random seed:", SEED)
print("Train images:", len(train))
print("Calibration images:", len(calib))
print("Test images:", len(test))
print("Total images:", len(image_names))
print("SHA-256 hashes saved to splits/sha256.txt")
print("Test filenames have not been displayed.")
print("\nSPLITS FROZEN!")
