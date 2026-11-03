
import random


def partition_mask_box(calibration_pool, n, seed):
    """
    Randomly divide calibration images into:

    mask_set: images with revealed masks
    box_only_set: images with hidden masks

    This function uses image names only.
    It never loads masks, boxes, or test data.
    """

    pool = list(calibration_pool)

    # Every image must be unique
    if len(pool) != len(set(pool)):
        raise ValueError("Duplicate images in calibration pool")

    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("n must be an integer")

    if not 0 <= n <= len(pool):
        raise ValueError("Invalid mask-set size")

    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer")

    # Sort for reproducible partitions
    pool = sorted(pool)

    # Independent random generator
    rng = random.Random(seed)
    rng.shuffle(pool)

    # First n images have revealed masks
    mask_set = sorted(pool[:n])

    # Remaining images have boxes only
    box_only_set = sorted(pool[n:])

    # Verify disjoint sets
    assert set(mask_set).isdisjoint(box_only_set)

    # Verify complete coverage
    assert (
        set(mask_set) | set(box_only_set)
        == set(calibration_pool)
    )

    return mask_set, box_only_set


def replication_seed(replication_id, base_seed=2026):
    """
    Generate a different seed for each replication.
    Replication IDs start at 0.
    """

    if not isinstance(replication_id, int):
        raise ValueError("Replication ID must be an integer")

    if replication_id < 0:
        raise ValueError("Replication ID cannot be negative")

    return base_seed + replication_id
