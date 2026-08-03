"""CPU tests for CUDA capability helpers (no GPU required)."""

from circuit_tracer.utils.cuda_info import (
    get_cuda_capabilities,
    has_cuda,
    has_vram_gb,
    primary_device_label,
    select_device,
)


def test_get_cuda_capabilities_shape():
    caps = get_cuda_capabilities()
    assert isinstance(caps.available, bool)
    assert caps.device_count >= 0
    payload = caps.to_dict()
    assert "available" in payload
    assert "devices" in payload
    assert has_cuda() == caps.available
    assert isinstance(primary_device_label(), str)


def test_select_device_prefers_cpu_when_requested():
    assert select_device("cpu").type == "cpu"


def test_has_vram_gb_false_when_no_cuda_or_insufficient():
    # Extremely high threshold should fail even on large GPUs
    assert has_vram_gb(10_000) is False
