"""Cell-line morphology scale constraints.

The public-data morphology training stores more than a class name: it records
the typical cell size for each line. Detection and tracking use these values to
reject biologically implausible duplicate centers, oversized/undersized partial
objects, and cells that enter or leave through the frame border.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def load_morphology_model(path: str) -> Dict[str, Any]:
    """Load ``morphology_model.json`` or a directory containing it."""
    model_path = path
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "morphology_model.json")
    with open(model_path, encoding="utf-8") as f:
        return json.load(f)


def constraints_for_cell_line(model: Dict[str, Any],
                              cell_line: str) -> Optional[Dict[str, float]]:
    """Return scale constraints for one trained cell line."""
    if not cell_line:
        return None
    stats = model.get("scale_stats", {})
    if cell_line in stats:
        return {str(k): float(v) for k, v in stats[cell_line].items()}

    wanted = cell_line.lower()
    for key, value in stats.items():
        if str(key).lower() == wanted:
            return {str(k): float(v) for k, v in value.items()}
    return None


def apply_constraints(detector=None, tracker=None,
                      constraints: Optional[Dict[str, float]] = None) -> None:
    """Apply scale constraints to detector and/or tracker when supported."""
    if not constraints:
        return
    if detector is not None and hasattr(detector, "set_morphology_constraints"):
        detector.set_morphology_constraints(**constraints)
    if tracker is not None and hasattr(tracker, "set_morphology_constraints"):
        tracker.set_morphology_constraints(**constraints)
