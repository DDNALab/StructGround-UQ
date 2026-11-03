
import numpy as np
from scipy import ndimage


def find_connected_components(mask):
    """
    Find separate damaged regions in a binary mask.
    Uses 8-connectivity, including diagonal neighbors.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")

    binary = mask.astype(bool)

    structure = np.ones((3, 3), dtype=int)

    labeled, count = ndimage.label(
        binary, structure=structure
    )

    return labeled, count


def create_tight_box(mask):
    """
    Return one tight box around ALL positive pixels.

    Format:
    (x_min, y_min, x_max, y_max)

    Maximum coordinates are exclusive.
    Returns None if the mask is empty.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")

    ys, xs = np.where(mask != 0)

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    y_min = int(ys.min())

    x_max = int(xs.max()) + 1
    y_max = int(ys.max()) + 1

    return (x_min, y_min, x_max, y_max)


def analyze_mask(mask):
    """
    Find connected components and a single
    enclosing box for the full class union.
    """
    labeled, count = find_connected_components(mask)

    box = create_tight_box(mask)

    return {
        "component_count": count,
        "bounding_box": box,
        "damaged_pixels": int(np.count_nonzero(mask))
    }
