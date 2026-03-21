# Challenge 2: Adversarial Perturbation Detection

## Objective

Images in the dataset contain adversarial perturbations hidden in localized regions. Your task is to detect and precisely localize the attacked areas by submitting a binary mask for each image.

## Rules

- Download perturbed images via `/task2/download/{image_id}`
- Analyze each image to find regions with adversarial modifications
- Submit a binary mask (PNG, grayscale: white = attacked region, black = clean) via `/task2/submit`
- Masks must be 224x224 pixels
- Resubmission is allowed — the best score per image is kept

## Scoring

Your predicted mask is compared to the ground truth using IoU (Intersection over Union), also known as the Jaccard index. The leaderboard ranks by mean IoU across all submitted images.

```
IoU = |Predicted ∩ Ground Truth| / |Predicted ∪ Ground Truth|

IoU = 1.0 → perfect detection
IoU = 0.0 → no overlap
```

## API Reference

**GET /task2/images**

```
Headers:
  Authorization: Bearer <jwt_token>

Response:
[
  {"image_id": "img_0000", "is_active": true},
  {"image_id": "img_0001", "is_active": true},
  ...
]
```

**GET /task2/download/{image_id}**

```
Headers:
  Authorization: Bearer <jwt_token>

Response: PNG image file
```

**POST /task2/submit**

```
Headers:
  Authorization: Bearer <jwt_token>

Body (multipart/form-data):
  mask: <file>           # Binary mask PNG (224x224, grayscale)
  image_id: "img_0001"

Response:
{
  "image_id": "img_0001",
  "iou_score": 0.723,
  "precision": 0.81,
  "recall": 0.65,
  "previous_best_iou": 0.690,
  "improved": true,
  "rank": 7
}
```
