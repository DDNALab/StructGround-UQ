
import sys
import unittest
from pathlib import Path

import numpy as np

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src" / "risk"))

from interval import compute_loss_interval


class TestLossInterval(unittest.TestCase):

    def test_case1_full_coverage(self):
        # Prediction covers the entire box
        U = np.ones((5, 5), dtype=np.uint8)
        box = (1, 1, 4, 4)

        result = compute_loss_interval(U, box)

        self.assertEqual(result["a"], 9)
        self.assertEqual(result["c"], 0)
        self.assertEqual(result["m"], 1)
        self.assertEqual(
            (result["lower_loss"], result["upper_loss"]),
            (0, 0)
        )

    def test_case2_no_coverage(self):
        # Prediction misses the entire box
        U = np.zeros((5, 5), dtype=np.uint8)
        box = (1, 1, 4, 4)

        result = compute_loss_interval(U, box)

        self.assertEqual(result["a"], 0)
        self.assertEqual(result["c"], 9)
        self.assertEqual(
            (result["lower_loss"], result["upper_loss"]),
            (1, 1)
        )

    def test_case3_partial_coverage(self):
        # Prediction covers part of the box
        U = np.zeros((5, 5), dtype=np.uint8)
        U[1:3, 1:3] = 1

        box = (1, 1, 4, 4)

        result = compute_loss_interval(U, box)

        self.assertEqual(result["a"], 4)
        self.assertEqual(result["c"], 5)
        self.assertEqual(
            (result["lower_loss"], result["upper_loss"]),
            (0, 1)
        )

    def test_true_loss_inside_interval(self):
        # True damage mask Y, used only for testing
        Y = np.zeros((5, 5), dtype=np.uint8)
        Y[1, 1] = 1
        Y[3, 3] = 1

        box = (1, 1, 4, 4)

        # Verify Y is nonempty and contained in B
        self.assertGreater(np.count_nonzero(Y), 0)

        box_mask = np.zeros_like(Y, dtype=bool)
        box_mask[1:4, 1:4] = True

        self.assertFalse(
            np.any((Y != 0) & ~box_mask)
        )

        predictions = []

        # Case 1: covers all true damage
        U1 = np.ones((5, 5), dtype=np.uint8)
        predictions.append(U1)

        # Case 2: misses all true damage
        U2 = np.zeros((5, 5), dtype=np.uint8)
        predictions.append(U2)

        # Case 3: covers some true damage
        U3 = np.zeros((5, 5), dtype=np.uint8)
        U3[1, 1] = 1
        predictions.append(U3)

        for U in predictions:
            result = compute_loss_interval(U, box)

            # True loss = 1 if any damage pixel
            # lies outside the predicted region.
            true_loss = int(
                np.any((Y != 0) & (U == 0))
            )

            self.assertLessEqual(
                result["lower_loss"], true_loss
            )
            self.assertLessEqual(
                true_loss, result["upper_loss"]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
