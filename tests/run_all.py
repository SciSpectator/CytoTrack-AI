#!/usr/bin/env python3
"""Fallback pytest-free test runner for CytoTrack AI.

Use this when `pytest` is unavailable. It discovers every test_*.py
file in the tests/ directory, invokes each test_* function, and reports
a summary.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import traceback
from dataclasses import dataclass
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402


@dataclass
class TestResult:
    name: str
    ok: bool
    detail: str = ""


def _make_fixtures():
    """Instantiate the fixtures normally provided by conftest.py."""
    from synthetic_data import SyntheticDataGenerator
    import cv2

    gen = SyntheticDataGenerator(width=480, height=320,
                                 num_cells=12, num_frames=8, seed=123)
    gen.generate_cells()
    frames, gts = [], []
    for i in range(gen.num_frames):
        img, gt = gen.generate_frame(i)
        frames.append(img)
        gts.append(gt)

    single_cell = np.full((80, 80, 3), 20, dtype=np.uint8)
    cv2.circle(single_cell, (40, 40), 14, (200, 200, 200), -1)
    cv2.circle(single_cell, (40, 40), 5, (120, 120, 120), -1)

    debris = np.full((80, 80, 3), 20, dtype=np.uint8)
    cv2.line(debris, (10, 40), (70, 42), (180, 180, 180), 2)

    return {
        "synthetic_gen": gen,
        "synthetic_frames": (frames, gts),
        "first_frame": frames[0],
        "single_cell_frame": single_cell,
        "debris_frame": debris,
    }


def _run_function(fn, fixtures):
    sig = inspect.signature(fn)
    kwargs = {}
    for name in sig.parameters:
        if name in fixtures:
            kwargs[name] = fixtures[name]
    fn(**kwargs)


def main() -> int:
    fixtures = _make_fixtures()
    modules = []
    for filename in sorted(os.listdir(_HERE)):
        if not filename.startswith("test_") or not filename.endswith(".py"):
            continue
        mod_name = "tests." + filename[:-3]
        try:
            modules.append(importlib.import_module(mod_name))
        except Exception as e:
            print(f"IMPORT-ERROR {filename}: {e}")
            traceback.print_exc()
            return 1

    results: List[TestResult] = []
    for module in modules:
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            try:
                _run_function(fn, fixtures)
                results.append(TestResult(f"{module.__name__}::{name}", True))
                print(f"ok     {module.__name__}::{name}")
            except AssertionError as e:
                results.append(TestResult(
                    f"{module.__name__}::{name}", False,
                    detail=f"AssertionError: {e}"))
                print(f"FAIL   {module.__name__}::{name}: {e}")
            except Exception as e:
                tb = traceback.format_exc().splitlines()
                results.append(TestResult(
                    f"{module.__name__}::{name}", False,
                    detail=f"{type(e).__name__}: {e}"))
                print(f"ERROR  {module.__name__}::{name}: {e}")
                for line in tb[-4:]:
                    print("        " + line)

    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed
    print("\n" + "=" * 60)
    print(f"Total: {len(results)}  Passed: {passed}  Failed: {failed}")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
