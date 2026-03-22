import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ml_security import Challenge1Workflow, Challenge2Workflow, Challenge3Workflow
import main as main_module
import ml_security.cli as cli_module
from solvers.solve_challenge_1 import solve as solve_challenge_1
from solvers.solve_challenge_2 import solve as solve_challenge_2


class FakeChallenge1Client:
    def __init__(self):
        self.queried = []
        self.submitted = []

    def query_image(self, image_path: str, image_id: str = "challenge_001") -> dict:
        self.queried.append((image_id, image_path))
        return {
            "predicted_class_name": "baseline",
            "target_probability": 0.2,
            "queries_remaining": 100,
        }

    def submit_adversarial(self, image_path: str, image_id: str = "challenge_001") -> dict:
        self.submitted.append((image_id, image_path))
        return {"target_probability": 0.4, "rank": 3}


class FakeChallenge3Client:
    def get_task3_info(self) -> dict:
        return {
            "features": [
                {"name": "f1", "dtype": "int", "min_value": 0, "max_value": 3},
            ],
            "query_budget": 10,
            "queries_used": 0,
            "test_set_size": 2,
        }

    def query_model(self, features: dict) -> dict:
        return {"prediction": int(features["f1"] > 0)}

    def get_test_features(self) -> dict:
        return {"csv": "f1\n0\n2\n"}

    def submit_predictions(self, csv_path: str) -> dict:
        rows = Path(csv_path).read_text().strip().splitlines()
        return {"submitted_rows": len(rows) - 1, "rank": 7}


class FakeChallenge2Client:
    def __init__(self):
        self.downloaded = []
        self.submitted = []

    def list_images(self):
        return [
            {"image_id": "active_image", "is_active": True},
            {"image_id": "inactive_image", "is_active": False},
        ]

    def download_image(self, image_id: str, save_path: str) -> str:
        self.downloaded.append((image_id, save_path))
        Path(save_path).write_bytes(b"fake-image")
        return save_path

    def submit_mask(self, mask_path: str, image_id: str) -> dict:
        self.submitted.append((image_id, mask_path))
        return {"image_id": image_id, "iou_score": 0.75}


class Challenge2WorkflowTests(unittest.TestCase):
    def test_run_processes_active_images_and_returns_structured_items(self):
        client = FakeChallenge2Client()

        def detector(image_path: str) -> np.ndarray:
            self.assertTrue(Path(image_path).exists())
            return np.ones((4, 4), dtype=np.uint8) * 255

        with TemporaryDirectory() as tmp_dir:
            workflow = Challenge2Workflow(client=client, output_dir=tmp_dir, detector=detector)

            result = workflow.run()

        self.assertEqual(len(client.downloaded), 1)
        self.assertEqual(client.downloaded[0][0], "active_image")
        self.assertEqual(len(client.submitted), 1)
        self.assertEqual(result.total_images, 2)
        self.assertEqual(result.processed_images, 1)
        self.assertEqual(result.skipped_images, 1)
        self.assertEqual(result.items[0].image_id, "active_image")
        self.assertEqual(result.items[0].submission["iou_score"], 0.75)


class Challenge1WorkflowTests(unittest.TestCase):
    def test_run_accepts_numpy_image_and_returns_structured_result(self):
        client = FakeChallenge1Client()
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        def attack(*, image_array: np.ndarray, baseline: dict, image_id: str):
            self.assertEqual(image_id, "challenge_001")
            self.assertEqual(baseline["target_probability"], 0.2)
            return {
                "best_image": image_array + 5,
                "history": [{"iteration": 1, "target_probability": 0.3}],
                "best_probability": 0.3,
            }

        with TemporaryDirectory() as tmp_dir:
            workflow = Challenge1Workflow(client=client, work_dir=tmp_dir, attack=attack)

            result = workflow.run(image)

        self.assertEqual(len(client.queried), 1)
        self.assertEqual(len(client.submitted), 1)
        self.assertEqual(result.baseline["predicted_class_name"], "baseline")
        self.assertEqual(result.best_probability, 0.3)
        self.assertEqual(result.submission["rank"], 3)
        self.assertEqual(result.history[0]["iteration"], 1)


class Challenge3WorkflowTests(unittest.TestCase):
    def test_run_collects_samples_and_submits_predictions(self):
        client = FakeChallenge3Client()
        samples = iter([{"f1": 0}, {"f1": 2}, {"f1": 1}])

        def sampler(features_info):
            self.assertEqual(features_info[0]["name"], "f1")
            return next(samples)

        def predictor(train_samples, test_rows, feature_names):
            self.assertEqual(feature_names, ["f1"])
            self.assertEqual(len(train_samples), 3)
            self.assertEqual(len(test_rows), 2)
            return [0, 1]

        with TemporaryDirectory() as tmp_dir:
            workflow = Challenge3Workflow(
                client=client,
                num_queries=3,
                work_dir=tmp_dir,
                sampler=sampler,
                predictor=predictor,
            )

            result = workflow.run()

        self.assertEqual(result.query_count, 3)
        self.assertEqual(result.training_samples[1]["features"]["f1"], 2)
        self.assertEqual(result.predictions, [0, 1])
        self.assertEqual(result.submission["submitted_rows"], 2)


class SolverCompatibilityTests(unittest.TestCase):
    def test_challenge_1_solver_uses_workflow_result(self):
        client = FakeChallenge1Client()
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        with TemporaryDirectory() as tmp_dir:
            result = solve_challenge_1(client, image, work_dir=tmp_dir)

        self.assertEqual(result.submission["rank"], 3)
        self.assertTrue(result.adversarial_path.endswith("adversarial.png"))

    def test_challenge_2_solver_returns_workflow_summary(self):
        client = FakeChallenge2Client()

        def detector(image_path: str) -> np.ndarray:
            return np.zeros((2, 2), dtype=np.uint8)

        with TemporaryDirectory() as tmp_dir:
            result = solve_challenge_2(client, output_dir=tmp_dir, detector=detector)

        self.assertEqual(result.processed_images, 1)
        self.assertEqual(result.items[0].image_id, "active_image")


class MainCliTests(unittest.TestCase):
    def test_main_prints_solver_result(self):
        fake_solver_module = type(
            "FakeSolverModule",
            (),
            {"solve": staticmethod(lambda client, output_dir: {"processed_images": 1, "output_dir": output_dir})},
        )

        with patch.object(cli_module, "ChallengeClient", return_value=object()):
            with patch.dict(sys.modules, {"solvers.solve_challenge_2": fake_solver_module}):
                with patch.object(sys, "argv", ["main.py", "--challenge", "2", "--api-key", "key"]):
                    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        main_module.main()

        self.assertIn("processed_images", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
