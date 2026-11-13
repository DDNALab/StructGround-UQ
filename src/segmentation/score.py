
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# Number of dacl10k output classes
NUM_CLASSES = 19

# Fixed preprocessing from the selected SegFormer
PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def score_image(image_path, model):
    """
    Return per-class segmentation scores for an image.

    image_path: Path to the input image.
    model: Already-loaded, frozen segmentation model.

    Output shape: (19, original_height, original_width)
    Output values: between 0 and 1.

    The function does not load checkpoint files
    or access segmentation ground-truth masks.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
        input_tensor = PREPROCESS(image).unsqueeze(0)

    # Detect model device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    input_tensor = input_tensor.to(device)

    # Disable training behavior and gradients
    model.eval()

    with torch.inference_mode():
        output = model(input_tensor)

        # Support a model returning either logits
        # directly or a Hugging Face output object.
        logits = (
            output.logits
            if hasattr(output, "logits")
            else output
        )

        if logits.ndim != 4:
            raise ValueError("Expected 4D model logits")

        if logits.shape[0] != 1:
            raise ValueError("Expected batch size 1")

        if logits.shape[1] != NUM_CLASSES:
            raise ValueError("Expected 19 output classes")

        # Restore original image resolution
        logits = F.interpolate(
            logits,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False
        )

        # Independent probabilities for each class
        scores = torch.sigmoid(logits)

        scores = (
            scores.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite score detected")

    return scores
