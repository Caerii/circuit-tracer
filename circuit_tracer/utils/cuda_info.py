"""CUDA capability detection for runtime and tests.

Provides a portable summary of available GPUs so callers can gate work by
VRAM (e.g. RTX 3080 ~10GB vs 24GB/32GB cards) without hard-coding device names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

__all__ = [
    "CudaDeviceInfo",
    "CudaCapabilities",
    "get_cuda_capabilities",
    "has_cuda",
    "has_vram_gb",
    "select_device",
    "primary_device_label",
]


@dataclass(frozen=True)
class CudaDeviceInfo:
    index: int
    name: str
    total_memory_gb: float
    major: int
    minor: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CudaCapabilities:
    available: bool
    device_count: int
    devices: tuple[CudaDeviceInfo, ...] = field(default_factory=tuple)
    primary_memory_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "deviceCount": self.device_count,
            "primaryMemoryGb": self.primary_memory_gb,
            "devices": [device.to_dict() for device in self.devices],
        }


def get_cuda_capabilities() -> CudaCapabilities:
    """Snapshot CUDA availability and per-device memory."""
    if not torch.cuda.is_available():
        return CudaCapabilities(available=False, device_count=0)

    devices: list[CudaDeviceInfo] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            CudaDeviceInfo(
                index=index,
                name=props.name,
                total_memory_gb=float(props.total_memory) / (1024**3),
                major=int(props.major),
                minor=int(props.minor),
            )
        )

    primary = devices[0].total_memory_gb if devices else 0.0
    return CudaCapabilities(
        available=True,
        device_count=len(devices),
        devices=tuple(devices),
        primary_memory_gb=primary,
    )


def has_cuda() -> bool:
    return get_cuda_capabilities().available


def has_vram_gb(min_gb: float, *, device_index: int = 0) -> bool:
    """Return True if the given CUDA device has at least *min_gb* of VRAM."""
    caps = get_cuda_capabilities()
    if not caps.available or device_index >= len(caps.devices):
        return False
    return caps.devices[device_index].total_memory_gb + 1e-6 >= float(min_gb)


def primary_device_label() -> str:
    """Human-readable primary GPU label for skip messages."""
    caps = get_cuda_capabilities()
    if not caps.available or not caps.devices:
        return "no CUDA device"
    device = caps.devices[0]
    return f"{device.name}, {device.total_memory_gb:.1f}GB"


def select_device(prefer: Literal["cuda", "cpu"] = "cuda") -> torch.device:
    """Select a torch device, optionally preferring CUDA when available."""
    if prefer == "cuda" and has_cuda():
        return torch.device("cuda")
    return torch.device("cpu")
