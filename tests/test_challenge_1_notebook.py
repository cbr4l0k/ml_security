import json
from pathlib import Path
import unittest


class Challenge1NotebookTests(unittest.TestCase):
    def test_notebook_contains_local_attack_pipeline_scaffold(self):
        notebook = json.loads(Path("notebooks/challenge_1.ipynb").read_text())
        sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("challenge_image_id =", sources)
        self.assertIn("random_probe_image", sources)
        self.assertIn("probe_submission = client.submit_adversarial", sources)
        self.assertIn('target_class = probe_submission["target_class"]', sources)
        self.assertIn("final_attack_path", sources)
        self.assertIn("class SimBADCTAttack", sources)
        self.assertIn("class SquareAttack", sources)
        self.assertIn("models = load_local_models()", sources)
        self.assertIn("run_adversarial_pipeline(", sources)
        self.assertIn("best_result", sources)
        self.assertIn("best_image_uint8", sources)
        self.assertIn("final_submission = client.submit_adversarial", sources)


if __name__ == "__main__":
    unittest.main()
