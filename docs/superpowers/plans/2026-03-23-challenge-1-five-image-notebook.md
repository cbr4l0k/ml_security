# Challenge 1 Five Image Notebook Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `notebooks/challenge_1.ipynb` into a starter experiment notebook that selects 5 simple dataset images, runs the challenge 1 workflow on each image, and summarizes the results for follow-up analysis.

**Architecture:** Keep the implementation notebook-local. The notebook will load a simple object-centric dataset, define a fixed five-image starter batch, run `Challenge1Workflow` in a loop, and assemble result records into a small analysis table and chart. Add one regression test that inspects the notebook JSON for the expected experiment scaffolding.

**Tech Stack:** Jupyter notebook JSON, Python stdlib `json`, pandas/matplotlib in notebook cells, existing `ml_security` workflow helpers

---

## Chunk 1: Regression Guard

### Task 1: Add a notebook content test

**Files:**
- Create: `tests/test_challenge_1_notebook.py`
- Test: `tests/test_challenge_1_notebook.py`

- [ ] **Step 1: Write the failing test**

Check that `notebooks/challenge_1.ipynb` contains dataset-loading, five-image selection, batch execution, and analysis markers.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_challenge_1_notebook.py -q`
Expected: FAIL because the current notebook does not include the new scaffolding.

- [ ] **Step 3: Write minimal implementation**

Update the notebook cells so those markers exist in the saved JSON.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_challenge_1_notebook.py -q`
Expected: PASS.

## Chunk 2: Notebook Refactor

### Task 2: Replace the minimal notebook with a starter experiment flow

**Files:**
- Modify: `notebooks/challenge_1.ipynb`

- [ ] **Step 1: Add setup and helper cells**

Include imports, plotting/dataframe helpers, work directory setup, budget check, and dataset-loading helpers.

- [ ] **Step 2: Add five-image starter batch selection**

Use fixed simple categories and define a `selected_samples` structure that the user can later extend.

- [ ] **Step 3: Add batch execution loop**

Instantiate `Challenge1Workflow`, run one experiment per selected image, and collect normalized result records.

- [ ] **Step 4: Add analysis cells**

Create a dataframe, rank the five runs, and draw a simple comparison chart.

## Chunk 3: Verification

### Task 3: Verify the notebook artifact

**Files:**
- Modify: `notebooks/challenge_1.ipynb`
- Test: `tests/test_challenge_1_notebook.py`

- [ ] **Step 1: Run the notebook content test**

Run: `python -m pytest tests/test_challenge_1_notebook.py -q`
Expected: PASS.

- [ ] **Step 2: Sanity-check the notebook JSON**

Run a small JSON parse command against `notebooks/challenge_1.ipynb`.
Expected: valid notebook JSON with the new cells present.
