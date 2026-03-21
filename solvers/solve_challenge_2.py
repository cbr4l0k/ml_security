"""Solver for Challenge 2: Adversarial Perturbation Detection.

Strategy:
  1. List all available images.
  2. Download each image.
  3. Analyze for adversarial perturbations (e.g., high-frequency analysis, denoising comparison).
  4. Generate a binary mask localizing the perturbed region.
  5. Submit the mask.
"""

import os

import numpy as np

from ml_security.client import ChallengeClient
from ml_security.utils import save_image


def detect_perturbation(image_path: str) -> np.ndarray:
    """Analyze an image and return a binary mask (224x224) of perturbed regions.

    TODO: Implement detection strategy. Possible approaches:
      - Compare image with a denoised version (median filter, bilateral filter)
      - High-pass filtering to detect localized high-frequency noise
      - Statistical analysis of pixel neighborhoods
      - Feature squeezing comparison
    """
    # Placeholder: return an empty mask (no perturbation detected)
    return np.zeros((224, 224), dtype=np.uint8)


def solve(client: ChallengeClient, output_dir: str = "task2_output") -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: List available images
    images = client.list_images()
    print(f"Found {len(images)} images")

    results = []
    for img_info in images:
        if not img_info.get("is_active", True):
            continue

        image_id = img_info["image_id"]
        image_path = os.path.join(output_dir, f"{image_id}.png")
        mask_path = os.path.join(output_dir, f"{image_id}_mask.png")

        # Step 2: Download the image
        client.download_image(image_id, image_path)

        # Step 3: Detect perturbation and generate mask
        mask = detect_perturbation(image_path)
        save_image(mask, mask_path)

        # Step 4: Submit the mask
        result = client.submit_mask(mask_path, image_id)
        print(f"  {image_id}: IoU={result['iou_score']:.3f}")
        results.append(result)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve Challenge 2: Perturbation Detection")
    parser.add_argument("--api-url", default="https://lms-dgunh.ru")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--output-dir", default="task2_output")
    args = parser.parse_args()

    c = ChallengeClient(args.api_url, args.api_key, args.username, args.password)
    solve(c, args.output_dir)
