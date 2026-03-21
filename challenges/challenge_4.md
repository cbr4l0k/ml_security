# Challenge 4: Backdoor Detection

## Objective

A fully connected neural network has been trained on the MNIST dataset (handwritten digits 0-9), but a hidden backdoor was injected during training. When a specific trigger pattern is present on the input image, the model always predicts a particular class regardless of the actual content. Your task is to download the model weights, analyze them, and determine: (1) the backdoor target class and (2) the trigger pattern.

## Rules

- Download model weights via `/task4/download-model` (.pt file for PyTorch)
- The network architecture is described at `/task4/info`
- Analyze the model by any means available (Neural Cleanse, weight analysis, etc.)
- Submit a trigger mask (PNG 28x28, grayscale) and the target class via `/task4/submit`
- Mask: white = trigger region, black = clean region

## Scoring

- **Target class (50%):** 0.5 points for correct class, 0 for incorrect
- **Trigger IoU (50%):** IoU between your mask and the ground truth trigger mask
- **Final score:** 0.5 * class_correct + 0.5 * trigger_iou (max 1.0)

## API Reference

**GET /task4/info**

```
Headers: none

Response:
{
  "task_description": "...",
  "model_architecture": "FCNet(784 -> 512 -> 256 -> 128 -> 10, ReLU)",
  "input_shape": [1, 28, 28],
  "num_classes": 10,
  "dataset": "MNIST"
}
```

**GET /task4/download-model**

```
Headers:
  Authorization: Bearer <jwt_token>

Response: backdoored_model.pt (PyTorch weights)
```

**POST /task4/submit**

```
Headers:
  Authorization: Bearer <jwt_token>

Body (multipart/form-data):
  trigger_mask: <file>    # PNG 28x28 (grayscale)
  target_class: 7          # Predicted backdoor target class

Response:
{
  "predicted_target_class": 7,
  "class_correct": true,
  "trigger_iou": 0.85,
  "combined_score": 0.925,
  "previous_best_score": 0.750,
  "improved": true,
  "rank": 2
}
```
