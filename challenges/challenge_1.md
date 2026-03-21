# Challenge 1: Adversarial Attack (Black-Box)

## Objective

You have access to an image classification neural network via API (black box). Given a source image and a target class, generate an adversarial perturbation that forces the model to classify the image as the target class with the highest possible confidence.

## Rules

- You can only interact with the model through the `/task1/query` endpoint
- Each query costs 1 budget unit (check remaining: `/task1/budget`)
- The API returns the predicted class and target class probability — no gradients, no logits
- Submit your adversarial image via `/task1/submit`

## Scoring

- **Primary metric:** target class probability (higher is better)
- **Secondary:** number of queries used (fewer is better)
- **Tertiary:** L2 norm of the perturbation (lower is better)

## API Reference

**POST /task1/query**

```
Headers:
  X-API-Key: your_api_key

Body (multipart/form-data):
  image: <file>              # PNG or JPEG image
  image_id: "challenge_001"  # Challenge identifier

Response:
{
  "predicted_class_id": 281,
  "predicted_class_name": "tabby cat",
  "predicted_probability": 0.7321,
  "target_probability": 0.0023,
  "queries_remaining": 9847
}
```

**POST /task1/submit**

```
Headers:
  Authorization: Bearer <jwt_token>

Body (multipart/form-data):
  adversarial_image: <file>  # Your adversarial image
  image_id: "challenge_001"

Response:
{
  "target_class": 390,
  "target_class_name": "eel",
  "target_probability": 0.847,
  "queries_used": 1523,
  "l2_norm": 12.45,
  "linf_norm": 0.031,
  "rank": 3
}
```
