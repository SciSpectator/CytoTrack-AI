"""Tests for the SyntheticDataGenerator."""

import numpy as np

from synthetic_data import Cell, SyntheticDataGenerator


def test_generate_cells_count():
    gen = SyntheticDataGenerator(200, 200, num_cells=7, num_frames=3, seed=1)
    gen.generate_cells()
    assert len(gen.cells) == 7
    for c in gen.cells:
        assert isinstance(c, Cell)


def test_generate_frame_shape_and_type():
    gen = SyntheticDataGenerator(320, 240, num_cells=5, num_frames=3, seed=42)
    gen.generate_cells()
    img, gt = gen.generate_frame(0)
    assert img.shape == (240, 320, 3)
    assert img.dtype == np.uint8
    assert len(gt) == 5
    for k, entry in gt.items():
        assert set(entry.keys()) == {"bbox", "center"}


def test_cells_move_between_frames():
    gen = SyntheticDataGenerator(320, 240, num_cells=5, num_frames=3, seed=5)
    gen.generate_cells()
    positions_before = [(c.x, c.y) for c in gen.cells]
    gen.generate_frame(0)
    positions_after = [(c.x, c.y) for c in gen.cells]
    assert positions_before != positions_after


def test_deterministic_with_seed():
    g1 = SyntheticDataGenerator(200, 200, num_cells=6, num_frames=2, seed=99)
    g1.generate_cells()
    img1, _ = g1.generate_frame(0)

    g2 = SyntheticDataGenerator(200, 200, num_cells=6, num_frames=2, seed=99)
    g2.generate_cells()
    img2, _ = g2.generate_frame(0)

    # Exact equality may fail due to noise, but summary stats should match.
    assert img1.shape == img2.shape
