
import sys
import unittest
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src" / "risk"))

from partition import (
    partition_mask_box,
    replication_seed
)


class TestPartition(unittest.TestCase):

    def setUp(self):
        self.pool = [
            f"image_{i}" for i in range(10)
        ]

    def test_disjoint_sets(self):
        masks, boxes = partition_mask_box(
            self.pool, n=3, seed=2026
        )

        self.assertTrue(
            set(masks).isdisjoint(boxes)
        )

    def test_complete_coverage(self):
        masks, boxes = partition_mask_box(
            self.pool, n=3, seed=2026
        )

        self.assertEqual(len(masks), 3)
        self.assertEqual(len(boxes), 7)

        self.assertEqual(
            set(masks) | set(boxes),
            set(self.pool)
        )

    def test_reproducibility(self):
        first = partition_mask_box(
            self.pool, n=3, seed=2026
        )

        second = partition_mask_box(
            self.pool, n=3, seed=2026
        )

        self.assertEqual(first, second)

    def test_different_replication_seeds(self):
        seeds = [
            replication_seed(i) for i in range(100)
        ]

        self.assertEqual(
            len(seeds), len(set(seeds))
        )

    def test_invalid_mask_count(self):
        with self.assertRaises(ValueError):
            partition_mask_box(
                self.pool, n=11, seed=2026
            )

    def test_duplicate_images(self):
        duplicate_pool = [
            "image_1", "image_1", "image_2"
        ]

        with self.assertRaises(ValueError):
            partition_mask_box(
                duplicate_pool, n=1, seed=2026
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
