
import argparse
import hashlib
import importlib
import os
import random
import sys
from pathlib import Path

# Set before importing torch for deterministic CUDA operations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch import nn


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "StructGround-UQ-data"

sys.path.insert(
    0, str(PROJECT / "src" / "segmentation")
)

from score import score_image

SEED = 2026
EXPECTED_COUNT = 1186

MODEL_HASH = (
    "166b778bdbc57ef2bfd5f71bc6ae724b"
    "f598d1a4c6a1b69afa4fbd936f2d965b"
)

CALIB_LIST = PROJECT / "splits" / "calib.txt"
IMAGE_DIR = DATA / "calib_images"
MODEL_PATH = DATA / "models" / "best_model_cpu.pth"

OUTPUT_DIR = (
    PROJECT / "results" / "scores" / "calib"
)


# --------------------------------------------------
# SegFormer wrapper
# --------------------------------------------------

# The publisher serialized this wrapper as
# __main__.SegModel.

class SegModel(nn.Module):

    def __init__(self, segformer):
        super().__init__()

        self.segformer = segformer

        self.upsample = nn.Upsample(
            scale_factor=4,
            mode="nearest"
        )

    def forward(self, x):
        return self.upsample(
            self.segformer(x).logits
        )


# --------------------------------------------------
# SHA-256 helper
# --------------------------------------------------

def sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# --------------------------------------------------
# Load frozen calibration list
# --------------------------------------------------

def load_calibration_list():

    names = [
        line.strip()
        for line in CALIB_LIST.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(names) != EXPECTED_COUNT:
        raise ValueError(
            "Unexpected calibration count"
        )

    if len(set(names)) != len(names):
        raise ValueError(
            "Duplicate calibration images"
        )

    return names


# --------------------------------------------------
# Preflight verification
# --------------------------------------------------

def preflight():

    names = load_calibration_list()

    expected = {
        name if name.lower().endswith(".jpg")
        else name + ".jpg"
        for name in names
    }

    actual = {
        path.name
        for path in IMAGE_DIR.glob("*.jpg")
    }

    if expected != actual:
        raise ValueError(
            "Calibration filenames do not match"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if sha256(MODEL_PATH) != MODEL_HASH:
        raise ValueError(
            "Checkpoint hash mismatch"
        )

    config = (
        PROJECT / "configs" / "scoring.yaml"
    ).read_text(encoding="utf-8")

    if "seed: 2026" not in config:
        raise ValueError(
            "Scoring seed mismatch"
        )

    print("PREFLIGHT PASSED")
    print("Calibration images:", len(names))
    print("Checkpoint SHA-256: MATCH")
    print("Random seed:", SEED)
    print("Test split accessed: No")

    return names


# --------------------------------------------------
# Allowed checkpoint global references
# --------------------------------------------------

ALLOWED_GLOBALS = """
transformers.models.segformer.modeling_segformer.SegformerDWConv
transformers.models.segformer.modeling_segformer.SegformerMixFFN
transformers.models.segformer.modeling_segformer.SegformerSelfOutput
transformers.models.segformer.modeling_segformer.SegformerEfficientSelfAttention
transformers.models.segformer.modeling_segformer.SegformerAttention
transformers.models.segformer.modeling_segformer.SegformerDropPath
transformers.models.segformer.modeling_segformer.SegformerDecodeHead
transformers.models.segformer.modeling_segformer.SegformerMLP
transformers.models.segformer.modeling_segformer.SegformerForSemanticSegmentation
transformers.models.segformer.modeling_segformer.SegformerLayer
transformers.models.segformer.modeling_segformer.SegformerOverlapPatchEmbeddings
transformers.models.segformer.modeling_segformer.SegformerModel
transformers.models.segformer.modeling_segformer.SegformerEncoder
transformers.models.segformer.configuration_segformer.SegformerConfig
transformers.activations.GELUActivation
torch.nn.modules.activation.ReLU
torch.nn.modules.linear.Identity
torch.nn.modules.container.ModuleList
torch.nn.modules.conv.Conv2d
torch.nn.modules.dropout.Dropout
torch.nn.modules.linear.Linear
torch.nn.modules.normalization.LayerNorm
torch.nn.modules.upsampling.Upsample
torch.nn.modules.batchnorm.BatchNorm2d
torch._C._nn.gelu
__main__.SegModel
""".split()


# --------------------------------------------------
# Resolve checkpoint classes
# --------------------------------------------------

def resolve_global(name):

    if name == "__main__.SegModel":
        return SegModel

    if name == "torch._C._nn.gelu":
        return torch._C._nn.gelu

    module_name, attribute = name.rsplit(
        ".", 1
    )

    module = importlib.import_module(
        module_name
    )

    return getattr(module, attribute)


# --------------------------------------------------
# Load frozen SegFormer model
# --------------------------------------------------

def load_model():

    observed = set(
        torch.serialization.get_unsafe_globals_in_checkpoint(
            str(MODEL_PATH)
        )
    )

    allowed = set(ALLOWED_GLOBALS)

    if not observed.issubset(allowed):
        raise ValueError(
            "Unexpected checkpoint global references: "
            + str(sorted(observed - allowed))
        )

    # The checkpoint stores the wrapper under __main__.SegModel.
    # When this module is imported by run_test_scoring.py, its real
    # import path is run_scoring.SegModel. Explicitly map the name.
    approved_types = [
        (SegModel, "__main__.SegModel")
        if name == "__main__.SegModel"
        else resolve_global(name)
        for name in sorted(observed)
    ]

    # Restricted checkpoint loading
    with torch.serialization.safe_globals(
        approved_types
    ):

        model = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

    if not isinstance(model, SegModel):
        raise TypeError(
            "Unexpected checkpoint model type"
        )

    # --------------------------------------------------
    # Compatibility fix for older SegFormer checkpoints
    # --------------------------------------------------
    #
    # Newer Transformers versions expect internal
    # configuration attributes that may be absent
    # from older serialized configurations.
    #
    # These changes happen in memory only.
    # The checkpoint and its weights remain frozen.

    for module in model.modules():

        config = getattr(
            module, "config", None
        )

        if config is None:
            continue

        if "_output_attentions" not in vars(config):
            config._output_attentions = False

        if "_attn_implementation_internal" not in vars(config):
            config._attn_implementation_internal = None

    print(
        "Legacy SegFormer configuration checked"
    )

    # --------------------------------------------------
    # Select GPU or CPU
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device).eval()

    return model, device


# --------------------------------------------------
# Run calibration scoring
# --------------------------------------------------

def run_scoring(names):

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model, device = load_model()

    print("Inference device:", device)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    for index, name in enumerate(
        names, start=1
    ):

        filename = (
            name
            if name.lower().endswith(".jpg")
            else name + ".jpg"
        )

        image_path = IMAGE_DIR / filename

        output_path = OUTPUT_DIR / (
            Path(filename).stem + ".npy"
        )

        # Reuse previously generated files
        # if scoring was interrupted.
        if output_path.exists():

            scores = np.load(
                output_path,
                allow_pickle=False
            )

        else:

            scores = score_image(
                image_path,
                model
            )

            temp_path = output_path.with_suffix(
                ".tmp"
            )

            with open(temp_path, "wb") as f:
                np.save(
                    f,
                    scores,
                    allow_pickle=False
                )

            os.replace(
                temp_path,
                output_path
            )

        # Validate score dimensions
        if (
            scores.ndim != 3
            or scores.shape[0] != 19
        ):
            raise ValueError(
                "Invalid score shape"
            )

        # Check for invalid numeric values
        if not np.isfinite(scores).all():
            raise ValueError(
                "Non-finite scores"
            )

        # Check probability range
        if (
            np.any(scores < 0)
            or np.any(scores > 1)
        ):
            raise ValueError(
                "Scores outside [0, 1]"
            )

        records.append(
            (
                output_path.name,
                sha256(output_path)
            )
        )

        if index % 50 == 0:
            print(
                f"Processed {index}/{len(names)}"
            )

    # --------------------------------------------------
    # Verify score file count
    # --------------------------------------------------

    actual_files = list(
        OUTPUT_DIR.glob("*.npy")
    )

    if len(actual_files) != len(names):
        raise ValueError(
            "Score file count mismatch"
        )

    # --------------------------------------------------
    # Save SHA-256 manifest
    # --------------------------------------------------

    manifest = (
        OUTPUT_DIR / "manifest_sha256.txt"
    )

    content = "".join(
        f"{digest}  {filename}\n"
        for filename, digest in sorted(records)
    )

    manifest.write_text(
        content,
        encoding="utf-8"
    )

    print("\nSCORING COMPLETED")
    print("Score files:", len(actual_files))
    print("Manifest:", manifest)


# --------------------------------------------------
# Main entry point
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        action="store_true",
        help="Run calibration inference"
    )

    parser.add_argument(
        "--g1-approved",
        action="store_true",
        help="Confirm PI checkpoint approval"
    )

    args = parser.parse_args()

    names = preflight()

    if not args.run:
        print(
            "Preflight only. No scores computed."
        )
        return

    if not args.g1_approved:
        raise SystemExit(
            "Gate G1 approval required before inference."
        )

    run_scoring(names)


if __name__ == "__main__":
    main()
