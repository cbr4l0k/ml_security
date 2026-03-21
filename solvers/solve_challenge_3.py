"""Solver for Challenge 3: Model Extraction.

Strategy:
  1. Get task info to understand features and budget.
  2. Generate strategic query samples (random sampling within feature ranges).
  3. Collect model responses to build a labeled dataset.
  4. Train a surrogate model (e.g., decision tree, logistic regression).
  5. Download test features, predict with the surrogate, submit CSV.
"""

import random

from ml_security.client import ChallengeClient
from ml_security.utils import parse_csv, save_predictions_csv


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


def solve(client: ChallengeClient, num_queries: int = 500) -> dict:
    # Step 1: Get task info
    info = client.get_task3_info()
    features_info = info["features"]
    budget = info["query_budget"]
    print(f"Features: {len(features_info)}, Budget: {budget}, Test size: {info['test_set_size']}")

    num_queries = min(num_queries, budget - info.get("queries_used", 0))

    # Step 2: Collect training data by querying the model
    train_x = []
    train_y = []
    for i in range(num_queries):
        sample = generate_random_sample(features_info)
        result = client.query_model(sample)
        train_x.append(sample)
        train_y.append(result["prediction"])
        if (i + 1) % 100 == 0:
            print(f"  Collected {i + 1}/{num_queries} samples")

    print(f"Training data: {len(train_x)} samples, "
          f"class balance: {sum(train_y)}/{len(train_y)} positive")

    # Step 3: Train a surrogate model
    # TODO: Implement surrogate training. Options:
    #   - scikit-learn DecisionTreeClassifier / LogisticRegression
    #   - Manual threshold-based rules from feature analysis
    #
    # Example with scikit-learn:
    # from sklearn.tree import DecisionTreeClassifier
    # feature_names = [f["name"] for f in features_info]
    # X = [[s[n] for n in feature_names] for s in train_x]
    # clf = DecisionTreeClassifier(max_depth=10)
    # clf.fit(X, train_y)

    # Step 4: Download test features and predict
    test_data = client.get_test_features()
    test_rows = parse_csv(test_data["csv"])
    print(f"Test set: {len(test_rows)} rows")

    # TODO: Replace with actual surrogate predictions
    # predictions = clf.predict([[float(row[n]) for n in feature_names] for row in test_rows])
    predictions = [0] * len(test_rows)  # Placeholder

    # Step 5: Submit predictions
    csv_path = "predictions.csv"
    save_predictions_csv(predictions, csv_path)
    result = client.submit_predictions(csv_path)
    print(f"Submission: fidelity={result['fidelity']:.3f}, "
          f"accuracy={result['accuracy']:.3f}, rank={result['rank']}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve Challenge 3: Model Extraction")
    parser.add_argument("--api-url", default="https://lms-dgunh.ru")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--num-queries", type=int, default=500)
    args = parser.parse_args()

    c = ChallengeClient(args.api_url, args.api_key, args.username, args.password)
    solve(c, args.num_queries)
