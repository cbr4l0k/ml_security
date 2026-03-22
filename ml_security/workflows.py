"""Notebook-friendly workflow helpers built on top of ``ChallengeClient``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ml_security.utils import load_image, parse_csv, save_image, save_predictions_csv


@dataclass(slots=True)
class Challenge1WorkflowResult:
    """Structured result for challenge 1."""

    baseline: dict
    best_probability: float
    submission: dict
    adversarial_path: str
    history: list[dict]


class Challenge1Workflow:
    """High-level workflow wrapper for challenge 1 notebook usage."""

    def __init__(
        self,
        client,
        image_id: str = "challenge_001",
        work_dir: str = ".",
        attack: Callable[..., dict] | None = None,
    ):
        self.client = client
        self.image_id = image_id
        self.work_dir = Path(work_dir)
        self.attack = attack or self._default_attack

    def run(self, image: str | np.ndarray) -> Challenge1WorkflowResult:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        source_path, image_array = self._prepare_image(image)
        baseline = self.client.query_image(source_path, self.image_id)
        attack_result = self.attack(
            image_array=image_array.copy(),
            baseline=baseline,
            image_id=self.image_id,
        )
        best_image = np.asarray(attack_result.get("best_image", image_array))
        adversarial_path = self.work_dir / "adversarial.png"
        save_image(best_image, str(adversarial_path))
        submission = self.client.submit_adversarial(str(adversarial_path), self.image_id)
        return Challenge1WorkflowResult(
            baseline=baseline,
            best_probability=float(attack_result.get("best_probability", baseline["target_probability"])),
            submission=submission,
            adversarial_path=str(adversarial_path),
            history=list(attack_result.get("history", [])),
        )

    def _prepare_image(self, image: str | np.ndarray) -> tuple[str, np.ndarray]:
        if isinstance(image, np.ndarray):
            source_path = self.work_dir / "source.png"
            save_image(image, str(source_path))
            return str(source_path), image
        return image, load_image(image)

    @staticmethod
    def _default_attack(*, image_array: np.ndarray, baseline: dict, image_id: str) -> dict:
        return {
            "best_image": image_array,
            "best_probability": baseline["target_probability"],
            "history": [],
        }


@dataclass(slots=True)
class Challenge2ItemResult:
    """Structured result for one challenge 2 image."""

    image_id: str
    image_path: str
    mask_path: str
    submission: dict


@dataclass(slots=True)
class Challenge2WorkflowResult:
    """Summary of a challenge 2 workflow run."""

    total_images: int
    processed_images: int
    skipped_images: int
    items: list[Challenge2ItemResult]

    def to_records(self) -> list[dict]:
        """Return notebook-friendly row data for tabular display."""
        return [
            {
                "image_id": item.image_id,
                "image_path": item.image_path,
                "mask_path": item.mask_path,
                **item.submission,
            }
            for item in self.items
        ]


class Challenge2Workflow:
    """High-level workflow wrapper for challenge 2 notebook usage."""

    def __init__(
        self,
        client,
        output_dir: str = "task2_output",
        detector: Callable[[str], np.ndarray] | None = None,
    ):
        self.client = client
        self.output_dir = Path(output_dir)
        self.detector = detector or self._default_detector

    def run(self) -> Challenge2WorkflowResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        images = self.client.list_images()
        items: list[Challenge2ItemResult] = []
        skipped = 0

        for img_info in images:
            if not img_info.get("is_active", True):
                skipped += 1
                continue

            image_id = img_info["image_id"]
            image_path = self.output_dir / f"{image_id}.png"
            mask_path = self.output_dir / f"{image_id}_mask.png"

            self.client.download_image(image_id, str(image_path))
            mask = self.detector(str(image_path))
            save_image(mask, str(mask_path))
            submission = self.client.submit_mask(str(mask_path), image_id)
            items.append(
                Challenge2ItemResult(
                    image_id=image_id,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    submission=submission,
                )
            )

        return Challenge2WorkflowResult(
            total_images=len(images),
            processed_images=len(items),
            skipped_images=skipped,
            items=items,
        )

    @staticmethod
    def _default_detector(image_path: str) -> np.ndarray:
        return np.zeros((224, 224), dtype=np.uint8)


@dataclass(slots=True)
class Challenge3WorkflowResult:
    """Structured result for challenge 3."""

    info: dict
    training_samples: list[dict]
    test_rows: list[dict[str, str]]
    predictions: list[int]
    submission: dict

    @property
    def query_count(self) -> int:
        return len(self.training_samples)


class Challenge3Workflow:
    """High-level workflow wrapper for challenge 3 notebook usage."""

    def __init__(
        self,
        client,
        num_queries: int = 500,
        work_dir: str = ".",
        sampler: Callable[[list[dict]], dict] | None = None,
        predictor: Callable[[list[dict], list[dict[str, str]], list[str]], list[int]] | None = None,
    ):
        self.client = client
        self.num_queries = num_queries
        self.work_dir = Path(work_dir)
        self.sampler = sampler or self._default_sampler
        self.predictor = predictor or self._default_predictor

    def run(self) -> Challenge3WorkflowResult:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        info = self.client.get_task3_info()
        features_info = info["features"]
        available_budget = info["query_budget"] - info.get("queries_used", 0)
        query_count = min(self.num_queries, available_budget)
        feature_names = [feature["name"] for feature in features_info]

        training_samples = []
        for _ in range(query_count):
            sample = self.sampler(features_info)
            prediction = self.client.query_model(sample)["prediction"]
            training_samples.append({"features": sample, "prediction": prediction})

        test_rows = parse_csv(self.client.get_test_features()["csv"])
        predictions = list(self.predictor(training_samples, test_rows, feature_names))
        csv_path = self.work_dir / "predictions.csv"
        save_predictions_csv(predictions, str(csv_path))
        submission = self.client.submit_predictions(str(csv_path))
        return Challenge3WorkflowResult(
            info=info,
            training_samples=training_samples,
            test_rows=test_rows,
            predictions=predictions,
            submission=submission,
        )

    @staticmethod
    def _default_sampler(features_info: list[dict]) -> dict:
        sample = {}
        for feat in features_info:
            name = feat["name"]
            min_val = feat.get("min_value", 0)
            max_val = feat.get("max_value", 100)
            if feat.get("dtype") == "int":
                sample[name] = int(min_val)
            else:
                sample[name] = float(min_val)
        return sample

    @staticmethod
    def _default_predictor(
        train_samples: list[dict],
        test_rows: list[dict[str, str]],
        feature_names: list[str],
    ) -> list[int]:
        return [0] * len(test_rows)


@dataclass(slots=True)
class Challenge4WorkflowResult:
    """Structured result for challenge 4."""

    info: dict
    model_path: str
    target_class: int
    trigger_mask_path: str
    submission: dict


class Challenge4Workflow:
    """High-level workflow wrapper for challenge 4 notebook usage."""

    def __init__(
        self,
        client,
        model_path: str = "backdoored_model.pt",
        work_dir: str = ".",
        analyzer: Callable[[str], tuple[int, np.ndarray]] | None = None,
    ):
        self.client = client
        self.model_path = model_path
        self.work_dir = Path(work_dir)
        self.analyzer = analyzer or self._default_analyzer

    def run(self) -> Challenge4WorkflowResult:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        info = self.client.get_task4_info()
        model_path = self.work_dir / self.model_path
        self.client.download_model(str(model_path))
        target_class, trigger_mask = self.analyzer(str(model_path))
        trigger_mask_path = self.work_dir / "trigger_mask.png"
        save_image(trigger_mask, str(trigger_mask_path))
        submission = self.client.submit_trigger(str(trigger_mask_path), target_class)
        return Challenge4WorkflowResult(
            info=info,
            model_path=str(model_path),
            target_class=target_class,
            trigger_mask_path=str(trigger_mask_path),
            submission=submission,
        )

    @staticmethod
    def _default_analyzer(model_path: str) -> tuple[int, np.ndarray]:
        return 0, np.zeros((28, 28), dtype=np.uint8)
