"""Environment report: interpreter, libraries, CUDA, external tools."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Any


def _tool_version(executable: str, argument: str = "--version") -> str | None:
    path = shutil.which(executable)
    if path is None:
        return None
    try:
        output = subprocess.run(
            [path, argument], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (output.stdout or output.stderr).strip().splitlines()
    return text[0] if text else None


def run_doctor() -> dict[str, Any]:
    import numpy
    import scipy
    import torch
    import yaml

    cuda_available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "ffmpeg": _tool_version("ffmpeg", "-version"),
        "ffprobe": _tool_version("ffprobe", "-version"),
        "typst": _tool_version("typst"),
        "node": _tool_version("node"),
        "npm": _tool_version("npm"),
        "pdftoppm": _tool_version("pdftoppm", "-v"),
    }
    # A CPU-only machine is fine for tests and the smoke run; the final
    # batched experiment expects CUDA, so its absence is only a warning.
    report["ok"] = True
    report["warnings"] = [] if cuda_available else ["CUDA unavailable: final runs need a GPU"]
    return report
