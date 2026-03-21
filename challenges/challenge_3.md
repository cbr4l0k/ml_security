# Challenge 3: Model Extraction

## Objective

You have black-box access to a credit scoring model via API. The model takes a loan application (a set of features) and returns a decision: approve (1) or deny (0). Your task is to extract the model's behavior by sending queries with various inputs, train a surrogate model locally, and submit predictions on a hidden test set.

## Rules

- Study the feature descriptions and example via `/task3/info`
- Query the model via `/task3/query` — budget is limited
- Download test features (unlabeled) via `/task3/test-features`
- Train a surrogate model on the collected data
- Submit a CSV with predictions via `/task3/submit`

## Scoring

- **Primary metric:** Fidelity — fraction of your predictions matching the original model
- **Secondary:** number of queries used (fewer is better)
- **Additional:** Accuracy — agreement with true labels

## API Reference

**GET /task3/info**

```
Headers:
  Authorization: Bearer <jwt_token>

Response:
{
  "task_description": "...",
  "features": [
    {"name": "age", "dtype": "int", "description": "...", "min_value": 18, "max_value": 75},
    ...
  ],
  "sample": {"age": 35, "income": 120.5, ..., "approved": 1},
  "query_budget": 2000,
  "queries_used": 0,
  "test_set_size": 500
}
```

**POST /task3/query**

```
Headers:
  Authorization: Bearer <jwt_token>

Body (JSON):
{
  "features": {
    "age": 35,
    "income": 120.5,
    "loan_amount": 500.0,
    ...
  }
}

Response:
{
  "prediction": 1,
  "probability": 0.8234,
  "queries_remaining": 1999
}
```

**GET /task3/test-features**

```
Headers:
  Authorization: Bearer <jwt_token>

Response:
{
  "columns": ["age", "income", ...],
  "num_rows": 500,
  "csv": "age,income,...\n35,120.5,...\n..."
}
```

**POST /task3/submit**

```
Headers:
  Authorization: Bearer <jwt_token>

Body (multipart/form-data):
  predictions_file: <CSV file>
  # Single column: prediction (0 or 1), one row per example

Response:
{
  "fidelity": 0.912,
  "accuracy": 0.874,
  "queries_used": 1523,
  "previous_best_fidelity": 0.890,
  "improved": true,
  "rank": 3
}
```
