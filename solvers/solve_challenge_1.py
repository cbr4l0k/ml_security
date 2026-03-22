"""Notebook-friendly challenge 1 workflow wrapper."""

import numpy as np

from ml_security.client import ChallengeClient
from ml_security.workflows import Challenge1Workflow, Challenge1WorkflowResult


def solve(
    client: ChallengeClient,
    image: str | np.ndarray,
    image_id: str = "challenge_001",
    work_dir: str = ".",
    attack=None,
) -> Challenge1WorkflowResult:
    """Run challenge 1 through the shared workflow layer."""
    workflow = Challenge1Workflow(
        client=client,
        image_id=image_id,
        work_dir=work_dir,
        attack=attack,
    )
    return workflow.run(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve Challenge 1: Black-box Adversarial Attack")
    parser.add_argument("--api-url", default="https://lms-dgunh.ru")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--image", required=True, help="Path to the source image")
    parser.add_argument("--image-id", default="challenge_001")
    parser.add_argument("--work-dir", default=".")
    args = parser.parse_args()

    c = ChallengeClient(args.api_url, args.api_key, args.username, args.password)
    result = solve(c, args.image, args.image_id, args.work_dir)
    print(result)
