
import sys
import unittest
from pathlib import Path

import numpy as np

# Import code from src/segmentation
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(project_root / "src" / "segmentation")
)

from union_mask import (
    combine_binary_masks,
    create_class_union_mask
)


class TestUnionMask(unittest.TestCase):

    def test_three_damaged_pixels(self):
        mask1 = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])

        mask2 = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])

        mask3 = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ])

        union = combine_binary_masks(
            [mask1, mask2, mask3]
        )

        self.assertEqual(int(union.sum()), 3)
        self.assertEqual(union.dtype, np.uint8)
        self.assertEqual(
            set(np.unique(union)),
            {0, 1}
        )

        print("\nCombined mask:")
        print(union)
        print("Damaged pixels:", int(union.sum()))

    def test_overlapping_masks(self):
        mask = np.array([
            [1, 0],
            [0, 0]
        ])

        union = combine_binary_masks([mask, mask])

        # Overlapping pixels count only once
        self.assertEqual(int(union.sum()), 1)

    def test_selected_class_only(self):
        sample = {
            "metadata": {
                "width": 5,
                "height": 5
            },
            "semseg_ground_truth_filled": {
                "polylines": [
                    {
                        "label": "Crack",
                        "filled": True,
                        "points": [[
                            [0.0, 0.0],
                            [0.2, 0.0],
                            [0.0, 0.2]
                        ]]
                    },
                    {
                        "label": "Rust",
                        "filled": True,
                        "points": [[
                            [0.6, 0.6],
                            [0.8, 0.6],
                            [0.6, 0.8]
                        ]]
                    }
                ]
            }
        }

        crack = create_class_union_mask(
            sample, "Crack"
        )
        rust = create_class_union_mask(
            sample, "Rust"
        )

        self.assertEqual(crack.shape, (5, 5))
        self.assertGreater(crack.sum(), 0)
        self.assertGreater(rust.sum(), 0)

        # Different class regions stay separate
        self.assertEqual(
            int(np.logical_and(crack, rust).sum()),
            0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
