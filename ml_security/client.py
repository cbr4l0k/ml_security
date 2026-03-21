"""API client for the ML Security challenge platform."""

import requests


class ChallengeClient:
    """Client for interacting with the lms-dgunh.ru ML security challenges."""

    def __init__(self, api_url: str, api_key: str, username: str | None = None, password: str | None = None):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.token: str | None = None

        if username and password:
            self.login(username, password)

    def login(self, username: str, password: str) -> str:
        resp = requests.post(
            f"{self.api_url}/auth/login",
            json={"username": username, "password": password},
        )
        resp.raise_for_status()
        self.token = resp.json()["access_token"]
        return self.token

    def _headers_api_key(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    def _headers_jwt(self) -> dict[str, str]:
        if not self.token:
            raise RuntimeError("Not authenticated. Call login() first or provide username/password.")
        return {"Authorization": f"Bearer {self.token}"}

    # ── Task 1: Black-box adversarial attack ──

    def query_image(self, image_path: str, image_id: str = "challenge_001") -> dict:
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{self.api_url}/task1/query",
                headers=self._headers_api_key(),
                files={"image": f},
                data={"image_id": image_id},
            )
        resp.raise_for_status()
        return resp.json()

    def submit_adversarial(self, image_path: str, image_id: str = "challenge_001") -> dict:
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{self.api_url}/task1/submit",
                headers=self._headers_jwt(),
                files={"adversarial_image": f},
                data={"image_id": image_id},
            )
        resp.raise_for_status()
        return resp.json()

    def get_budget(self) -> dict:
        resp = requests.get(
            f"{self.api_url}/task1/budget",
            headers=self._headers_api_key(),
        )
        resp.raise_for_status()
        return resp.json()

    # ── Task 2: Adversarial perturbation detection ──

    def list_images(self) -> list[dict]:
        resp = requests.get(
            f"{self.api_url}/task2/images",
            headers=self._headers_jwt(),
        )
        resp.raise_for_status()
        return resp.json()

    def download_image(self, image_id: str, save_path: str) -> str:
        resp = requests.get(
            f"{self.api_url}/task2/download/{image_id}",
            headers=self._headers_jwt(),
        )
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return save_path

    def submit_mask(self, mask_path: str, image_id: str) -> dict:
        with open(mask_path, "rb") as f:
            resp = requests.post(
                f"{self.api_url}/task2/submit",
                headers=self._headers_jwt(),
                files={"mask": f},
                data={"image_id": image_id},
            )
        resp.raise_for_status()
        return resp.json()

    # ── Task 3: Model extraction ──

    def get_task3_info(self) -> dict:
        resp = requests.get(
            f"{self.api_url}/task3/info",
            headers=self._headers_jwt(),
        )
        resp.raise_for_status()
        return resp.json()

    def query_model(self, features: dict) -> dict:
        resp = requests.post(
            f"{self.api_url}/task3/query",
            headers=self._headers_jwt(),
            json={"features": features},
        )
        resp.raise_for_status()
        return resp.json()

    def get_test_features(self) -> dict:
        resp = requests.get(
            f"{self.api_url}/task3/test-features",
            headers=self._headers_jwt(),
        )
        resp.raise_for_status()
        return resp.json()

    def submit_predictions(self, csv_path: str) -> dict:
        with open(csv_path, "rb") as f:
            resp = requests.post(
                f"{self.api_url}/task3/submit",
                headers=self._headers_jwt(),
                files={"predictions_file": f},
            )
        resp.raise_for_status()
        return resp.json()

    # ── Task 4: Backdoor detection ──

    def get_task4_info(self) -> dict:
        resp = requests.get(f"{self.api_url}/task4/info")
        resp.raise_for_status()
        return resp.json()

    def download_model(self, save_path: str = "backdoored_model.pt") -> str:
        resp = requests.get(
            f"{self.api_url}/task4/download-model",
            headers=self._headers_jwt(),
        )
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return save_path

    def submit_trigger(self, mask_path: str, target_class: int) -> dict:
        with open(mask_path, "rb") as f:
            resp = requests.post(
                f"{self.api_url}/task4/submit",
                headers=self._headers_jwt(),
                files={"trigger_mask": f},
                data={"target_class": str(target_class)},
            )
        resp.raise_for_status()
        return resp.json()
