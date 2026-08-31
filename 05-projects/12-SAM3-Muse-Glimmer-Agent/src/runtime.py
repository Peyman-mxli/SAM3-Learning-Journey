"""Runtime diagnostics and reproducibility metadata."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import Any


def collect_runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "validated_inference": False,
    }
    try:
        import torch
        metadata.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available() else None,
        })
    except ImportError:
        metadata["torch"] = None
        metadata["cuda_available"] = False
    return metadata
