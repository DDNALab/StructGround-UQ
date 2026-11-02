
import numpy as np
from PIL import Image, ImageDraw


def combine_binary_masks(masks):
    """
    Combine multiple binary masks into one union mask.
    All masks must have the same dimensions.
    """
    if not masks:
        raise ValueError("At least one mask is required")

    masks = [np.asarray(m) for m in masks]

    shape = masks[0].shape

    if len(shape) != 2:
        raise ValueError("Masks must be 2D")

    if any(m.shape != shape for m in masks):
        raise ValueError("Mask dimensions must match")

    union = np.zeros(shape, dtype=np.uint8)

    for mask in masks:
        union |= (mask != 0).astype(np.uint8)

    return union


def create_class_union_mask(sample, class_name):
    """
    Build one binary semantic mask for a class using
    an image's Hugging Face dacl10k polygon records.

    1 = selected class
    0 = background or another class
    """
    metadata = sample["metadata"]
    width = int(metadata["width"])
    height = int(metadata["height"])

    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")

    annotations = sample.get(
        "semseg_ground_truth_filled"
    )

    if annotations is None:
        raise ValueError(
            "Ground-truth annotations are unavailable"
        )

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    for polygon in annotations.get("polylines", []):
        if polygon.get("label") != class_name:
            continue

        if not polygon.get("filled", False):
            continue

        for ring in polygon.get("points", []):
            if len(ring) < 3:
                continue

            coordinates = [
                (round(x * width), round(y * height))
                for x, y in ring
            ]

            # Draw all polygons of the selected class
            draw.polygon(coordinates, fill=1)

    return np.array(canvas, dtype=np.uint8)
