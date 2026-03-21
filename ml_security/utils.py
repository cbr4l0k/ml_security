"""Shared utilities for image I/O, mask creation, and CSV handling."""

import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image file and return it as a numpy array (H, W, C) or (H, W)."""
    img = Image.open(path)
    return np.array(img)


def save_image(array: np.ndarray, path: str) -> str:
    """Save a numpy array as an image file. Returns the path."""
    img = Image.fromarray(array.astype(np.uint8))
    img.save(path)
    return path


def create_binary_mask(shape: tuple[int, int], regions: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Create a binary mask with white rectangles at the given regions.

    Args:
        shape: (height, width) of the mask.
        regions: list of (y_start, x_start, y_end, x_end) rectangles to mark as white (255).

    Returns:
        Grayscale numpy array (H, W) with 0 for clean and 255 for marked regions.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for y0, x0, y1, x1 in regions:
        mask[y0:y1, x0:x1] = 255
    return mask


def parse_csv(csv_string: str) -> list[dict[str, str]]:
    """Parse a CSV string into a list of dicts (one per row)."""
    reader = csv.DictReader(io.StringIO(csv_string))
    return list(reader)


def save_predictions_csv(predictions: list[int], path: str) -> str:
    """Save a list of integer predictions to a single-column CSV file."""
    p = Path(path)
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prediction"])
        for pred in predictions:
            writer.writerow([pred])
    return str(p)
