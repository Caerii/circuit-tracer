"""Lightweight CUDA smoke tests sized for ~8–10GB GPUs (e.g. RTX 3080).

Excluded from default CI via ``requires_gpu`` / ``slow`` markers.
"""

from __future__ import annotations

import gc

import pytest
import torch

from circuit_tracer import ReplacementModel

pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.vram_10gb,
    pytest.mark.slow,
    pytest.mark.requires_disk,
]


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="module")
def gemma_tl_bf16():
    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "gemma",
        dtype=torch.bfloat16,
        backend="transformerlens",
    )
    yield model
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_feature_intervention_smoke(gemma_tl_bf16):
    model = gemma_tl_bf16
    prompt = "The National Digital Analytics Group (ND"
    interventions = [(21, 7, 5066, 0.0)]
    logits, activations = model.feature_intervention(
        prompt,
        interventions,
        return_activations=False,
    )
    assert activations is None
    assert logits.ndim == 3
    assert logits.shape[-1] > 0


def test_feature_intervention_generate_logit_cache_smoke(gemma_tl_bf16):
    """Locks empty-list logit cache init used by feature_intervention_generate."""
    model = gemma_tl_bf16
    prompt = "The capital of France is"
    interventions = [(21, 7, 5066, 0.0)]
    text, logits, activations = model.feature_intervention_generate(
        prompt,
        interventions,
        return_activations=False,
        max_new_tokens=2,
        do_sample=False,
        verbose=False,
    )
    assert isinstance(text, str)
    assert activations is None
    assert logits.ndim == 2, "generate must return 2-D logits (seq, vocab)"
    assert logits.shape[0] >= 1
    assert not torch.isnan(logits).any()
