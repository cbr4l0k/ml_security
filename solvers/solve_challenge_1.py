"""Solver for Challenge 1: Black-box Adversarial Attack.

Strategy:
  1. Query the original image to get baseline prediction and target probability.
  2. Iteratively perturb the image (e.g., random noise, coordinate-wise estimation).
  3. Keep the best perturbation that maximizes target class probability.
  4. Submit the best adversarial image.
"""

import numpy as np

from ml_security.client import ChallengeClient
from ml_security.utils import load_image, save_image


def solve(client: ChallengeClient, image_path: str, image_id: str = "challenge_001") -> dict:
    # Step 1: Query the original image for baseline
    baseline = client.query_image(image_path, image_id)
    print(f"Baseline: class={baseline['predicted_class_name']}, "
          f"target_prob={baseline['target_probability']:.4f}, "
          f"budget={baseline['queries_remaining']}")

    original = load_image(image_path).astype(np.float64)
    best_image = original.copy()
    best_prob = baseline["target_probability"]

    # Step 2: Iterative perturbation loop
    # TODO: Implement attack strategy (e.g., SimBA, Square Attack, NES gradient estimation)
    #
    # Example skeleton for a random-search approach:
    # for iteration in range(max_iterations):
    #     perturbation = generate_perturbation(original.shape)
    #     candidate = np.clip(original + perturbation, 0, 255)
    #     save_image(candidate, "candidate.png")
    #     result = client.query_image("candidate.png", image_id)
    #     if result["target_probability"] > best_prob:
    #         best_prob = result["target_probability"]
    #         best_image = candidate
    #         print(f"  Improved: target_prob={best_prob:.4f} at iteration {iteration}")
    #     if result["queries_remaining"] <= 0:
    #         break

    print(f"Best target probability achieved: {best_prob:.4f}")

    # Step 3: Submit the best adversarial image
    adversarial_path = "adversarial.png"
    save_image(best_image, adversarial_path)
    result = client.submit_adversarial(adversarial_path, image_id)
    print(f"Submission: target_prob={result['target_probability']:.4f}, "
          f"rank={result['rank']}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve Challenge 1: Black-box Adversarial Attack")
    parser.add_argument("--api-url", default="https://lms-dgunh.ru")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--image", required=True, help="Path to the source image")
    parser.add_argument("--image-id", default="challenge_001")
    args = parser.parse_args()

    c = ChallengeClient(args.api_url, args.api_key, args.username, args.password)
    solve(c, args.image, args.image_id)
