"""Shared pytest fixtures for CytoTrack AI tests."""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from synthetic_data import SyntheticDataGenerator  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_gen():
    gen = SyntheticDataGenerator(width=480, height=320,
                                 num_cells=12, num_frames=8, seed=123)
    gen.generate_cells()
    return gen


@pytest.fixture(scope="session")
def synthetic_frames(synthetic_gen):
    frames, gts = [], []
    for i in range(synthetic_gen.num_frames):
        img, gt = synthetic_gen.generate_frame(i)
        frames.append(img)
        gts.append(gt)
    return frames, gts


@pytest.fixture
def first_frame(synthetic_frames):
    frames, _ = synthetic_frames
    return frames[0]


@pytest.fixture
def single_cell_frame():
    img = np.full((80, 80, 3), 20, dtype=np.uint8)
    import cv2
    cv2.circle(img, (40, 40), 14, (200, 200, 200), -1)
    cv2.circle(img, (40, 40), 5, (120, 120, 120), -1)
    return img


@pytest.fixture
def debris_frame():
    img = np.full((80, 80, 3), 20, dtype=np.uint8)
    import cv2
    # elongated streak, obviously not a round cell
    cv2.line(img, (10, 40), (70, 42), (180, 180, 180), 2)
    return img
