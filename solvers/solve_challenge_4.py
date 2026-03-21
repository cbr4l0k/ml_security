"""Solver for Challenge 4: Backdoor Detection.

Strategy:
  1. Get model architecture info.
  2. Download the backdoored model weights.
  3. Analyze the model to find the trigger pattern and target class.
     Possible approaches: Neural Cleanse, weight analysis, activation clustering.
  4. Submit the trigger mask and target class.
"""

import numpy as np

from ml_security.client import ChallengeClient
from ml_security.utils import save_image


def analyze_model(model_path: str) -> tuple[int, np.ndarray]:
    """Analyze a backdoored model to find the target class and trigger mask.

    Args:
        model_path: Path to the .pt model weights file.

    Returns:
        (target_class, trigger_mask) where trigger_mask is a 28x28 grayscale array.

    TODO: Implement analysis. Possible approaches:
      - Neural Cleanse: optimize a minimal perturbation per class, the class with
        the smallest perturbation norm is likely the backdoor target.
      - Weight analysis: inspect the final layer weights for anomalous patterns.
      - Activation clustering: cluster activations on clean data, look for outliers.

    Example with PyTorch:
      import torch
      state_dict = torch.load(model_path, map_location="cpu")
      # Build FCNet(784 -> 512 -> 256 -> 128 -> 10, ReLU)
      # Run Neural Cleanse optimization for each class...
    """
    # Placeholder: guess class 0 with an empty mask
    target_class = 0
    trigger_mask = np.zeros((28, 28), dtype=np.uint8)
    return target_class, trigger_mask


def solve(client: ChallengeClient, model_path: str = "backdoored_model.pt") -> dict:
    # Step 1: Get model info
    info = client.get_task4_info()
    print(f"Architecture: {info['model_architecture']}")
    print(f"Input shape: {info['input_shape']}, Classes: {info['num_classes']}")

    # Step 2: Download the model
    client.download_model(model_path)
    print(f"Model saved to {model_path}")

    # Step 3: Analyze the model
    target_class, trigger_mask = analyze_model(model_path)
    print(f"Detected target class: {target_class}")

    mask_path = "trigger_mask.png"
    save_image(trigger_mask, mask_path)

    # Step 4: Submit
    result = client.submit_trigger(mask_path, target_class)
    print(f"Submission: class_correct={result['class_correct']}, "
          f"trigger_iou={result['trigger_iou']:.3f}, "
          f"score={result['combined_score']:.3f}, rank={result['rank']}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve Challenge 4: Backdoor Detection")
    parser.add_argument("--api-url", default="https://lms-dgunh.ru")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--model-path", default="backdoored_model.pt")
    args = parser.parse_args()

    c = ChallengeClient(args.api_url, args.api_key, args.username, args.password)
    solve(c, args.model_path)
