
import sys
import unittest
from pathlib import Path

import numpy as np

project = Path(__file__).resolve().parents[1]

sys.path.insert(
    0, str(project / "src" / "segmentation")
)

from bounding_boxes import (
    find_connected_components,
    create_tight_box,
    analyze_mask
)


class TestBoundingBoxes(unittest.TestCase):

    def test_two_opposite_corner_blobs(self):

        mask = np.array([
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1]
        ], dtype=np.uint8)

        result = analyze_mask(mask)

        print("\nTEST MASK:")
        print(mask)

        print("\nConnected components:",
              result["component_count"])

        print("Bounding box:",
              result["bounding_box"])

        print("Damaged pixels:",
              result["damaged_pixels"])

        self.assertEqual(
            result["component_count"], 2
        )

        self.assertEqual(
            result["bounding_box"], (0, 0, 5, 5)
        )

        self.assertEqual(
            result["damaged_pixels"], 6
        )

    def test_single_blob(self):

        mask = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])

        labeled, count = find_connected_components(mask)

        self.assertEqual(count, 1)

        self.assertEqual(
            create_tight_box(mask), (1, 1, 3, 3)
        )

    def test_empty_mask(self):

        mask = np.zeros((5, 5), dtype=np.uint8)

        result = analyze_mask(mask)

        self.assertEqual(result["component_count"], 0)
        self.assertIsNone(result["bounding_box"])
        self.assertEqual(result["damaged_pixels"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
