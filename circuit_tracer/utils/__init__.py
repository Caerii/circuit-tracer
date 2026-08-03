import torch

from circuit_tracer.utils.create_graph_files import create_graph_files as create_graph_files
from circuit_tracer.utils.cuda_info import (
    get_cuda_capabilities,
    has_cuda,
    has_vram_gb,
    select_device,
)


def get_default_device() -> torch.device:
    """Get the default device, preferring CUDA if available."""
    return select_device("cuda")


__all__ = [
    "create_graph_files",
    "get_default_device",
    "get_cuda_capabilities",
    "has_cuda",
    "has_vram_gb",
    "select_device",
]
