# `ml-security`

`ml-security` is now organized for notebook-first challenge work.

The low-level transport layer stays in `ChallengeClient`, and the main interactive experience lives in per-challenge workflow helpers:

- `Challenge1Workflow`
- `Challenge2Workflow`
- `Challenge3Workflow`
- `Challenge4Workflow`

These workflows are designed for iterative notebook use:

- they return structured result objects instead of relying on `print(...)`
- they accept workflow callbacks so you can keep your attack, detector, sampler, or analyzer logic in notebook cells
- they preserve the existing client for direct API access when you need lower-level control

## Install

Base install:

```bash
uv sync
```

Notebook-oriented install:

```bash
uv sync --extra notebook
```

Optional ML dependencies:

```bash
uv sync --extra ml
```

## Environment File

Create a local `.env` from the example template:

```bash
cp .env.example .env
```

Then fill in:

```dotenv
ML_SECURITY_API_URL=https://lms-dgunh.ru
ML_SECURITY_API_KEY=...
ML_SECURITY_USERNAME=...
ML_SECURITY_PASSWORD=...
```

The package now supports:

```python
from ml_security import ChallengeClient

client = ChallengeClient.from_env()
```

Environment variables already exported in your shell take precedence over values in `.env`.

## Jupyter Setup

Install a project kernel tied to `.venv`:

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV "$(pwd)/.venv" --name=project
```

Start JupyterLab:

```bash
uv run --with jupyter jupyter lab
```

Then choose the `project` kernel inside the notebook UI.

## Notebook Quickstart

```python
import numpy as np

from ml_security import Challenge1Workflow, ChallengeClient

client = ChallengeClient.from_env()

image = np.zeros((224, 224, 3), dtype=np.uint8)

def attack(*, image_array, baseline, image_id):
    return {
        "best_image": image_array,
        "best_probability": baseline["target_probability"],
        "history": [{"step": "baseline", "target_probability": baseline["target_probability"]}],
    }

workflow = Challenge1Workflow(client=client, work_dir="notebook_runs/challenge1", attack=attack)
result = workflow.run(image)

result
```

Challenge 2 is the clearest example of the new notebook style:

```python
import numpy as np

from ml_security import Challenge2Workflow

def detector(image_path: str) -> np.ndarray:
    return np.zeros((224, 224), dtype=np.uint8)

workflow = Challenge2Workflow(client=client, output_dir="notebook_runs/challenge2", detector=detector)
result = workflow.run()
result.to_records()
```

## API Shape

### Low-level

`ChallengeClient` still provides direct HTTP wrappers such as:

- `query_image`
- `list_images`
- `query_model`
- `download_model`

### Notebook-first

The workflow layer wraps the challenge loops and returns structured results:

- `Challenge1Workflow.run(image)`
- `Challenge2Workflow.run()`
- `Challenge3Workflow.run()`
- `Challenge4Workflow.run()`

The legacy `solvers/solve_challenge_*.py` modules remain available, but they now delegate to the shared workflow layer instead of maintaining separate script-only logic.

## Example Notebook

See [notebooks/ml-security-notebook-quickstart.ipynb](notebooks/ml-security-notebook-quickstart.ipynb) for a notebook-oriented walkthrough.
