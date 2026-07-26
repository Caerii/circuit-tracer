"""Shared utilities for both NNSight and TransformerLens replacement models."""

import warnings
from collections.abc import Sequence

import torch

# Type definition for an intervention tuple (layer, position, feature_idx, value)
Intervention = tuple[
    int | torch.Tensor,
    int | slice | torch.Tensor,
    int | torch.Tensor,
    int | float | torch.Tensor,
]


def ensure_tokenized(
    prompt: str | torch.Tensor | list[int],
    tokenizer,
    device: torch.device | str | None,
    model_name: str,
) -> torch.Tensor:
    """Convert prompt to 1-D tensor of token ids with proper special token handling.

    This function ensures that a special token (BOS/PAD) is prepended to the input
    sequence. The first token position in transformer models typically exhibits
    unusually high norm and an excessive number of active features due to how models
    process the beginning of sequences. By prepending a special token, we ensure that
    actual content tokens have more consistent and interpretable feature activations,
    avoiding the artifacts present at position 0. This prepended token is later
    ignored during attribution analysis.

    Args:
        prompt: String, tensor, or list of token ids representing a single sequence
        tokenizer: HuggingFace-compatible tokenizer
        device: Target device for the output tensor
        model_name: Name of the model (used for model-specific handling)

    Returns:
        1-D tensor of token ids with BOS/PAD token at the beginning

    Raises:
        TypeError: If prompt is not str, tensor, or list
        ValueError: If tensor has wrong shape (must be 1-D or 2-D with batch size 1)
    """

    if isinstance(prompt, str):
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(
            0
        )
    elif isinstance(prompt, torch.Tensor):
        tokens = prompt.squeeze()
    elif isinstance(prompt, list):
        tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
    else:
        raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    if tokens.ndim > 1:
        raise ValueError(f"Tensor must be 1-D, got shape {tokens.shape}")

    tokens = tokens.to(device)

    gemma_3_it = "gemma-3" in model_name and model_name.endswith("-it")
    if gemma_3_it:
        ignore_prefix = torch.tensor([2, 105, 2364, 107], dtype=tokens.dtype, device=tokens.device)
        tokenization_error = (
            "Input tokens should start with <bos><start_of_turn>user\n, but got {tokens}"
        )
        assert tokens.size(0) >= 4 and torch.all(tokens[:4] == ignore_prefix), (
            tokenization_error.format(tokens=tokenizer.decode(tokens.cpu().tolist()))
        )
        return tokens

    # Check if a special token is already present at the beginning
    if tokens[0] in tokenizer.all_special_ids:
        return tokens

    # Prepend a special token to avoid artifacts at position 0
    candidate_bos_token_ids = [
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
    ]
    candidate_bos_token_ids += tokenizer.all_special_ids

    dummy_bos_token_id = next(filter(None, candidate_bos_token_ids))
    if dummy_bos_token_id is None:
        warnings.warn(
            "No suitable special token found for BOS token replacement. "
            "The first token will be ignored."
        )
    else:
        tokens = torch.cat([torch.tensor([dummy_bos_token_id], device=tokens.device), tokens])

    return tokens.to(device)


def convert_open_ended_interventions(
    interventions: Sequence[Intervention],
) -> list[Intervention]:
    """Convert open-ended interventions into position-0 equivalents.

    An intervention is *open-ended* if its position component is a ``slice`` whose
    ``stop`` attribute is ``None`` (e.g. ``slice(1, None)``). Such interventions will
    also apply to tokens generated in an open-ended generation loop. In such cases,
    when use_past_kv_cache=True, the model only runs the most recent token
    (and there is thus only 1 position).
    """
    converted = []
    for layer, pos, feature_idx, value in interventions:
        if isinstance(pos, slice) and pos.stop is None:
            converted.append((layer, 0, feature_idx, value))
    return converted
