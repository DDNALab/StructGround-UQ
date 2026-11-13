
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

project = Path(__file__).resolve().parents[1]

sys.path.insert(
    0, str(project / "src" / "segmentation")
)

from score import score_image


class DummySegmentationModel(torch.nn.Module):
    """
    Deterministic test model with 19 output channels.
    This is NOT the trained SegFormer checkpoint.
    """

    def forward(self, x):
        # Create low-resolution logits from image data
        x = F.avg_pool2d(x[:, :1], kernel_size=4)

        # Return the required 19 channels
        return x.repeat(1, 19, 1, 1)


class TestScore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_path = (
            Path(self.temp_dir.name) / "test_image.png"
        )

        # Create a reproducible test image
        image = np.zeros(
            (31, 47, 3), dtype=np.uint8
        )

        image[:, :, 0] = 100
        image[:, :, 1] = 150
        image[:, :, 2] = 200

        Image.fromarray(image).save(self.image_path)

        self.model = DummySegmentationModel()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_output_shape(self):
        scores = score_image(
            self.image_path, self.model
        )

        self.assertEqual(scores.shape, (19, 31, 47))

    def test_scores_between_zero_and_one(self):
        scores = score_image(
            self.image_path, self.model
        )

        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    def test_determinism(self):
        first = score_image(
            self.image_path, self.model
        )

        second = score_image(
            self.image_path, self.model
        )

        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
