import numpy as np

from ml_security.client import ChallengeClient
from ml_security.workflows import Challenge2Workflow, Challenge2WorkflowResult


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


def solve(
    client: ChallengeClient,
    output_dir: str = "task2_output",
    detector=detect_perturbation,
) -> Challenge2WorkflowResult:
    """Run challenge 2 through the shared workflow layer."""
    workflow = Challenge2Workflow(
        client=client,
        output_dir=output_dir,
        detector=detector,
    )
    return workflow.run()


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
    result = solve(c, args.output_dir)
    print(result)
