"""GPT-2 Small loading and residual-stream activation helpers.

The default intervention point is ``blocks.6.hook_resid_pre``. TransformerLens
uses zero-indexed block numbers, so this activation is the residual stream
entering block 6 (and therefore the residual stream after block 5). It is a
``resid_pre`` hook, not the ``resid_post`` output of block 6.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from src.utils import resolve_device, resolve_dtype


DEFAULT_HOOK_NAME = "blocks.6.hook_resid_pre"


def load_model(
    model_name: str = "gpt2-small",
    device: str | None = None,
    dtype: str | torch.dtype = torch.float32,
) -> HookedTransformer:
    """Load a pretrained TransformerLens model in evaluation mode.

    When ``device`` is omitted, CUDA is selected if available and CPU
    otherwise. TransformerLens creates the model directly on that device.
    ``center_writing_weights=True`` matches the model-processing override
    registered by SAELens for the project's ``gpt2-small-res-jb`` checkpoint.
    """
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype)
    model = HookedTransformer.from_pretrained(
        model_name,
        device=str(resolved_device),
        dtype=resolved_dtype,
        center_writing_weights=True,
    )
    # TransformerLens consumes this preprocessing option while constructing the
    # model, but not every release stores it on ``model.cfg``.  Preserve the
    # actual load provenance so the SAE compatibility gate can verify it rather
    # than reporting the known setting as "unrecognized".
    model._steering_denoiser_load_kwargs = {  # type: ignore[attr-defined]
        "center_writing_weights": True,
    }
    model.eval()
    return model


def _validate_tokens(model: HookedTransformer, tokens: Tensor) -> None:
    """Validate the token batch without moving or casting it implicitly."""
    assert tokens.ndim == 2, (
        f"Expected tokens with shape [batch, seq], got {tuple(tokens.shape)}."
    )
    assert tokens.dtype == torch.long, (
        f"Expected torch.long tokens, got {tokens.dtype}."
    )

    model_device = next(model.parameters()).device
    assert tokens.device == model_device, (
        f"Tokens are on {tokens.device}, but the model is on {model_device}. "
        "Move tokens explicitly before calling this function."
    )


def _validate_hook_name(model: HookedTransformer, hook_name: str) -> None:
    """Fail early when an intervention hook is not present in the model."""
    assert hook_name in model.hook_dict, f"Unknown TransformerLens hook: {hook_name}"


@torch.inference_mode()
def get_residual_activations(
    model: HookedTransformer,
    tokens: Tensor,
    hook_name: str = DEFAULT_HOOK_NAME,
    **forward_kwargs: object,
) -> Tensor:
    """Return a residual-stream activation with shape ``[batch, seq, d_model]``.

    The returned tensor retains the model activation's dtype and device.
    ``tokens`` must already be on the model's device.
    """
    _validate_tokens(model, tokens)
    _validate_hook_name(model, hook_name)

    _, cache = model.run_with_cache(
        tokens,
        names_filter=hook_name,
        return_type=None,
        **forward_kwargs,
    )
    activations = cache[hook_name]

    expected_shape = (tokens.shape[0], tokens.shape[1], model.cfg.d_model)
    assert activations.shape == expected_shape, (
        f"Expected residual activations with shape {expected_shape}, "
        f"got {tuple(activations.shape)}."
    )
    return activations


def get_logits_with_intervention(
    model: HookedTransformer,
    tokens: Tensor,
    hook_name: str,
    intervention_fn: Callable[[Tensor], Tensor],
    **forward_kwargs: object,
) -> Tensor:
    """Run the model while replacing the activation at ``hook_name``.

    ``intervention_fn`` receives the batched hook activation and must return a
    tensor with exactly the same shape, dtype, and device. This function does
    not disable autograd so it can later support training a differentiable
    denoiser through the frozen language model. Wrap the call in
    ``torch.no_grad()`` for inference-only use.
    """
    _validate_tokens(model, tokens)
    _validate_hook_name(model, hook_name)
    hook_was_called = False

    def replace_activation(
        activation: Tensor,
        _hook: object | None = None,
        **_: object,
    ) -> Tensor:
        """Accept both TransformerLens positional and keyword hook calls.

        TransformerLens releases differ slightly here: current versions call
        ``hook(activation, hook=self)``, while lightweight test doubles and
        older releases may pass the hook object positionally.
        """
        nonlocal hook_was_called
        hook_was_called = True
        modified = intervention_fn(activation)
        assert isinstance(modified, Tensor), (
            "intervention_fn must return a torch.Tensor."
        )
        assert modified.shape == activation.shape, (
            "intervention_fn changed the activation shape from "
            f"{tuple(activation.shape)} to {tuple(modified.shape)}."
        )
        assert modified.dtype == activation.dtype, (
            f"intervention_fn changed dtype from {activation.dtype} "
            f"to {modified.dtype}."
        )
        assert modified.device == activation.device, (
            f"intervention_fn changed device from {activation.device} "
            f"to {modified.device}."
        )
        return modified

    logits = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=[(hook_name, replace_activation)],
        **forward_kwargs,
    )
    assert hook_was_called, f"Intervention hook was not called: {hook_name}"
    assert isinstance(logits, Tensor)
    return logits


@torch.inference_mode()
def get_clean_logits(
    model: HookedTransformer,
    tokens: Tensor,
    **forward_kwargs: object,
) -> Tensor:
    """Return clean next-token logits for a batch of tokens."""
    _validate_tokens(model, tokens)
    logits = model(tokens, return_type="logits", **forward_kwargs)
    assert isinstance(logits, Tensor)
    return logits


@torch.inference_mode()
def sanity_check_identity_intervention(
    model: HookedTransformer,
    hook_name: str = DEFAULT_HOOK_NAME,
) -> None:
    """Check that an identity hook reproduces clean logits for two strings."""
    tokens = model.to_tokens(
        [
            "The sky is blue.",
            "Mechanistic interpretability studies neural networks.",
        ],
        prepend_bos=True,
    )
    clean_logits = get_clean_logits(model, tokens)
    identity_logits = get_logits_with_intervention(
        model=model,
        tokens=tokens,
        hook_name=hook_name,
        intervention_fn=lambda activation: activation,
    )
    torch.testing.assert_close(identity_logits, clean_logits)
