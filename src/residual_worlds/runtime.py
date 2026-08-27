"""Process-level runtime configuration.

Torch's default CPU thread pool (one thread per logical core) is
counterproductive for this project's many small-tensor operations and
oversubscribes shared machines. The scientific batches are either small
(structural geometry, scalar references) or GPU-resident, so a small
fixed CPU pool is both faster and a better neighbor. Override with
``RW_TORCH_CPU_THREADS`` if a dedicated machine is available.
"""

from __future__ import annotations

import os

_configured = False


def configure_torch_cpu(default_threads: int = 2) -> None:
    """Cap the intra-op CPU thread pool once per process (idempotent)."""
    global _configured
    if _configured:
        return
    import torch

    threads = int(os.environ.get("RW_TORCH_CPU_THREADS", default_threads))
    torch.set_num_threads(max(1, threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Interop pool can only be set before first parallel work; if a
        # library call beat us to it, the intra-op cap still applies.
        pass
    _configured = True
