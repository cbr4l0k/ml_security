import numpy as np

from ml_security.client import ChallengeClient
from ml_security.workflows import Challenge4Workflow, Challenge4WorkflowResult


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


def solve(
    client: ChallengeClient,
    model_path: str = "backdoored_model.pt",
    work_dir: str = ".",
    analyzer=analyze_model,
) -> Challenge4WorkflowResult:
    """Run challenge 4 through the shared workflow layer."""
    workflow = Challenge4Workflow(
        client=client,
        model_path=model_path,
        work_dir=work_dir,
        analyzer=analyzer,
    )
    return workflow.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve Challenge 4: Backdoor Detection")
    parser.add_argument("--api-url", default="https://lms-dgunh.ru")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--model-path", default="backdoored_model.pt")
    parser.add_argument("--work-dir", default=".")
    args = parser.parse_args()

    c = ChallengeClient(args.api_url, args.api_key, args.username, args.password)
    result = solve(c, args.model_path, args.work_dir)
    print(result)
