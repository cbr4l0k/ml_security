import random

from ml_security.client import ChallengeClient
from ml_security.workflows import Challenge3Workflow, Challenge3WorkflowResult


def generate_random_sample(features_info: list[dict]) -> dict:
    """Generate a random feature vector based on feature descriptions."""
    sample = {}
    for feat in features_info:
        name = feat["name"]
        min_val = feat.get("min_value", 0)
        max_val = feat.get("max_value", 100)
        if feat.get("dtype") == "int":
            sample[name] = random.randint(int(min_val), int(max_val))
        else:
            sample[name] = round(random.uniform(float(min_val), float(max_val)), 2)
    return sample


def solve(
    client: ChallengeClient,
    num_queries: int = 500,
    work_dir: str = ".",
    sampler=generate_random_sample,
    predictor=None,
) -> Challenge3WorkflowResult:
    """Run challenge 3 through the shared workflow layer."""
    workflow = Challenge3Workflow(
        client=client,
        num_queries=num_queries,
        work_dir=work_dir,
        sampler=sampler,
        predictor=predictor,
    )
    return workflow.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve Challenge 3: Model Extraction")
    parser.add_argument("--api-url", default="https://lms-dgunh.ru")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--work-dir", default=".")
    args = parser.parse_args()

    c = ChallengeClient(args.api_url, args.api_key, args.username, args.password)
    result = solve(c, args.num_queries, args.work_dir)
    print(result)
