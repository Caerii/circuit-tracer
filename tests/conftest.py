import pytest
import torch

from circuit_tracer.utils.cuda_info import get_cuda_capabilities, has_vram_gb, primary_device_label

# Backward-compatible aliases used by existing tests
has_32gb = has_vram_gb(32)
has_24gb = has_vram_gb(24)
has_10gb = has_vram_gb(10)
has_8gb = has_vram_gb(8)

_VRAM_MARKERS = {
    "vram_8gb": 8,
    "vram_10gb": 10,
    "vram_24gb": 24,
    "vram_32gb": 32,
}


@pytest.fixture(autouse=True)
def set_torch_seed() -> None:
    torch.manual_seed(42)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_gpu: marks tests requiring a CUDA GPU")
    config.addinivalue_line("markers", "vram_8gb: requires >=8GB CUDA VRAM")
    config.addinivalue_line("markers", "vram_10gb: requires >=10GB CUDA VRAM")
    config.addinivalue_line("markers", "vram_24gb: requires >=24GB CUDA VRAM")
    config.addinivalue_line("markers", "vram_32gb: requires >=32GB CUDA VRAM")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    caps = get_cuda_capabilities()
    label = primary_device_label()

    skip_no_cuda = pytest.mark.skip(reason=f"CUDA not available ({label})")
    for item in items:
        needs_gpu = "requires_gpu" in item.keywords or any(
            mark in item.keywords for mark in _VRAM_MARKERS
        )
        if needs_gpu and not caps.available:
            item.add_marker(skip_no_cuda)
            continue

        for mark_name, min_gb in _VRAM_MARKERS.items():
            if mark_name in item.keywords and not has_vram_gb(min_gb):
                item.add_marker(
                    pytest.mark.skip(reason=f"Requires >={min_gb}GB VRAM (detected {label})")
                )
