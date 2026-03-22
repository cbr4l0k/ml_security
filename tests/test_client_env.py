import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ml_security.client import ChallengeClient


class ChallengeClientEnvTests(unittest.TestCase):
    def test_from_env_loads_credentials_from_dotenv_file(self):
        env_text = "\n".join(
            [
                "ML_SECURITY_API_URL=https://example.test",
                "ML_SECURITY_API_KEY=file-key",
                "ML_SECURITY_USERNAME=file-user",
                "ML_SECURITY_PASSWORD=file-pass",
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(env_text)

            with patch.object(ChallengeClient, "login", return_value="jwt-token") as login:
                client = ChallengeClient.from_env(env_path=env_path)

        self.assertEqual(client.api_url, "https://example.test")
        self.assertEqual(client.api_key, "file-key")
        login.assert_called_once_with("file-user", "file-pass")

    def test_from_env_prefers_existing_environment_variables(self):
        env_text = "\n".join(
            [
                "ML_SECURITY_API_URL=https://example.test",
                "ML_SECURITY_API_KEY=file-key",
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(env_text)

            with patch.dict(os.environ, {"ML_SECURITY_API_KEY": "env-key"}, clear=False):
                client = ChallengeClient.from_env(env_path=env_path)

        self.assertEqual(client.api_key, "env-key")

    def test_from_env_requires_api_key(self):
        with TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("")

            with self.assertRaisesRegex(ValueError, "ML_SECURITY_API_KEY"):
                ChallengeClient.from_env(env_path=env_path)


if __name__ == "__main__":
    unittest.main()
