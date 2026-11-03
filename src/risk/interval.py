
import numpy as np


def compute_loss_interval(predicted_region, box):
    """
    Compute lower and upper missed-damage loss bounds.

    predicted_region: 2D binary prediction U
    box: (x_min, y_min, x_max, y_max)

    Box maximum coordinates are exclusive.

    Assumes the true mask is nonempty and
    fully contained inside the bounding box.
    """

    U = np.asarray(predicted_region)

    if U.ndim != 2:
        raise ValueError("Predicted region must be 2D")

    height, width = U.shape

    x_min, y_min, x_max, y_max = map(int, box)

    if not (
        0 <= x_min < x_max <= width
        and 0 <= y_min < y_max <= height
    ):
        raise ValueError("Invalid bounding box")

    # Pixels predicted inside box B
    inside_box = U[y_min:y_max, x_min:x_max] != 0

    # Total pixels in B
    box_area = (x_max - x_min) * (y_max - y_min)

    # a = pixels inside both U and B
    a = int(np.count_nonzero(inside_box))

    # c = pixels in B but outside U
    c = box_area - a

    # Minimum possible true mask size
    m = 1

    # Mathematical interval rules
    lower_loss = 1 if a == 0 else 0
    upper_loss = 1 if c > 0 else 0

    return {
        "a": a,
        "c": c,
        "m": m,
        "lower_loss": lower_loss,
        "upper_loss": upper_loss
    }
