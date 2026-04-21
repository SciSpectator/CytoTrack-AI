"""
CytoTrack AI - Hardware profiler / latency tuner
=================================================

Auto-detects local GPU (VRAM), system RAM, and CPU count, then assigns a
tier (low / mid / high / extreme) and a set of runtime knobs the rest of
the pipeline can consult.

Design rule
-----------
These knobs only trade *latency* vs *throughput*.  They must NEVER change
model inputs, model weights, classifier resolution, detector thresholds
that affect recall, DSPy reasoning strategy, or any other decision that
alters analytical accuracy.  In other words: weaker hardware runs the
same model with the same inputs, just less in parallel / less often.

Public API
----------
HardwareProfile                       dataclass with tier + knobs
detect_hardware() -> HardwareProfile  one-shot probe, cached
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# -------------------------------------------------------------- dataclass ---
@dataclass
class HardwareProfile:
    """
    Snapshot of local compute resources and the latency/throughput knobs
    downstream code should honour.
    """
    tier: str = "mid"                  # "low" | "mid" | "high" | "extreme"
    cpu_count: int = 1
    ram_gb: float = 4.0
    has_cuda: bool = False
    gpu_name: str = ""
    vram_gb: float = 0.0
    reason: str = ""                   # human-readable explanation

    # --- latency/throughput knobs (safe to vary; don't affect accuracy) ---
    # Number of CPU worker threads for parallel strategies / dataloaders.
    num_workers: int = 2
    # Classifier inference batch size (model itself is unchanged).
    classifier_batch_size: int = 8
    # Preview redraw framerate in the Qt preview dialog.
    preview_fps: int = 10
    # Video writer output fps (encoding cadence, not frame content).
    video_fps: int = 10
    # Enable/skip the expensive-but-optional strategies in the detector
    # fusion pipeline. Core strategies (adaptive threshold, Otsu,
    # watershed) always stay on; only the extras are gated.
    use_blob_detector: bool = True
    use_hough_circles: bool = True
    # Skip factor for the lost-cell-recovery ReAct reasoning. The cheaper
    # classical NCC path always runs; this only gates the LLM fallback.
    enable_recovery_llm: bool = False
    # Preferred torch device for classifier inference.
    torch_device: str = "cpu"

    # Extra info rows the UI can show
    notes: List[str] = field(default_factory=list)

    # ---------------------------------------------------------- helpers ----
    def summary(self) -> str:
        """One-line human summary (suitable for logs / GUI status)."""
        gpu = f"{self.gpu_name} ({self.vram_gb:.1f} GB VRAM)" \
              if self.has_cuda else "CPU only"
        return (f"Tier: {self.tier.upper()}  \u2022  {gpu}  \u2022  "
                f"{self.cpu_count} cores  \u2022  {self.ram_gb:.1f} GB RAM")

    def long_description(self) -> str:
        lines = [self.summary(), ""]
        lines.append(f"Latency tuning: {self.reason}")
        lines.append("")
        lines.append("Runtime knobs (accuracy preserved):")
        lines.append(f"  \u2022 workers: {self.num_workers}")
        lines.append(f"  \u2022 classifier batch: {self.classifier_batch_size}")
        lines.append(f"  \u2022 preview fps: {self.preview_fps}")
        lines.append(f"  \u2022 video fps: {self.video_fps}")
        lines.append(f"  \u2022 blob detector strategy: "
                     f"{'on' if self.use_blob_detector else 'off'}")
        lines.append(f"  \u2022 hough circles strategy: "
                     f"{'on' if self.use_hough_circles else 'off'}")
        lines.append(f"  \u2022 LLM lost-cell recovery: "
                     f"{'on' if self.enable_recovery_llm else 'off'}")
        lines.append(f"  \u2022 torch device: {self.torch_device}")
        for n in self.notes:
            lines.append(f"  \u2022 {n}")
        return "\n".join(lines)


# ------------------------------------------------------ resource probes ----
def _probe_ram_gb() -> float:
    """Read /proc/meminfo first; fall back to psutil; default 4 GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        pass
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 4.0


def _probe_gpu_via_torch() -> Tuple[bool, str, float]:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            idx = 0
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            vram = props.total_memory / (1024 ** 3)
            return True, name, vram
    except Exception:
        pass
    return False, "", 0.0


def _probe_gpu_via_nvidia_smi() -> Tuple[bool, str, float]:
    """Last-resort: parse nvidia-smi output."""
    if not shutil.which("nvidia-smi"):
        return False, "", 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3)
        line = out.strip().splitlines()[0]
        name, mem = [p.strip() for p in line.split(",", 1)]
        return True, name, float(mem) / 1024.0
    except Exception:
        return False, "", 0.0


# ---------------------------------------------------------- tier policy ----
def _assign_tier(has_cuda: bool, vram_gb: float, ram_gb: float,
                 cpu_count: int) -> Tuple[str, str]:
    if has_cuda and vram_gb >= 16:
        return "extreme", "High-VRAM GPU detected — run every strategy in parallel."
    if has_cuda and vram_gb >= 8:
        return "high", "Desktop-class GPU — full strategy fusion, batch inference."
    if has_cuda and vram_gb >= 4:
        return "mid", "Entry GPU / laptop — keep full fusion, smaller batches."
    if ram_gb >= 12 and cpu_count >= 8:
        return "mid", "No GPU but strong CPU — CPU inference, full fusion."
    if ram_gb >= 6 and cpu_count >= 4:
        return "low", ("Modest CPU-only machine — skip optional detector "
                       "strategies to keep latency manageable.")
    return "low", ("Limited resources — skip optional detector strategies "
                   "and shrink batch/preview rates. Model accuracy is "
                   "unchanged.")


def _knobs_for_tier(tier: str, has_cuda: bool, vram_gb: float,
                    cpu_count: int) -> dict:
    """
    Return a dict of knobs for the given tier.

    Accuracy-critical knobs (model, weights, input size, decision thresholds)
    are intentionally absent. Only latency/throughput dials are tuned.
    """
    if tier == "extreme":
        return dict(num_workers=min(16, max(4, cpu_count)),
                    classifier_batch_size=64,
                    preview_fps=30, video_fps=15,
                    use_blob_detector=True, use_hough_circles=True,
                    enable_recovery_llm=True,
                    torch_device="cuda")
    if tier == "high":
        return dict(num_workers=min(12, max(4, cpu_count)),
                    classifier_batch_size=32,
                    preview_fps=24, video_fps=12,
                    use_blob_detector=True, use_hough_circles=True,
                    enable_recovery_llm=True,
                    torch_device="cuda")
    if tier == "mid":
        return dict(num_workers=min(8, max(2, cpu_count // 2)),
                    classifier_batch_size=16 if has_cuda else 8,
                    preview_fps=15, video_fps=10,
                    use_blob_detector=True, use_hough_circles=True,
                    enable_recovery_llm=has_cuda,
                    torch_device="cuda" if has_cuda else "cpu")
    # "low" -> shed latency without touching model behaviour.
    return dict(num_workers=max(1, min(4, cpu_count // 2 or 1)),
                classifier_batch_size=4,
                preview_fps=8, video_fps=8,
                use_blob_detector=False,    # heaviest, least critical
                use_hough_circles=False,    # heaviest, least critical
                enable_recovery_llm=False,
                torch_device="cpu")


# ----------------------------------------------------------- public API ----
_CACHED: Optional[HardwareProfile] = None


def detect_hardware(force: bool = False) -> HardwareProfile:
    """
    Probe the local machine and return a HardwareProfile. Cached; call
    with force=True to re-probe.
    """
    global _CACHED
    if _CACHED is not None and not force:
        return _CACHED

    cpu_count = os.cpu_count() or 1
    ram_gb = _probe_ram_gb()
    has_cuda, gpu_name, vram_gb = _probe_gpu_via_torch()
    if not has_cuda:
        has_cuda, gpu_name, vram_gb = _probe_gpu_via_nvidia_smi()

    tier, reason = _assign_tier(has_cuda, vram_gb, ram_gb, cpu_count)
    knobs = _knobs_for_tier(tier, has_cuda, vram_gb, cpu_count)

    notes: List[str] = []
    if has_cuda and vram_gb < 2:
        notes.append("GPU has <2 GB VRAM; classifier stays on CPU for safety.")
        knobs["torch_device"] = "cpu"
    if ram_gb < 4:
        notes.append("Low RAM — preview cache limited to fewer frames.")

    prof = HardwareProfile(
        tier=tier,
        cpu_count=cpu_count,
        ram_gb=round(ram_gb, 2),
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        vram_gb=round(vram_gb, 2),
        reason=reason,
        notes=notes,
        **knobs,
    )
    _CACHED = prof
    return prof


if __name__ == "__main__":
    p = detect_hardware()
    print(p.long_description())
