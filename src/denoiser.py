"""Small residual MLP denoiser for GPT-2 residual activations."""

import torch
from torch import Tensor, nn


class ResidualDenoiser(nn.Module):
    """Residual MLP with optional scalar noise-level conditioning.

    The final projection is initialized to zero, so ``forward(x) == x`` at
    initialization. Arbitrary leading dimensions are preserved.
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 768,
        condition_on_noise: bool = False,
        conditioning_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        assert d_model > 0 and hidden_dim > 0 and conditioning_hidden_dim > 0
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.condition_on_noise = condition_on_noise
        self.conditioning_hidden_dim = conditioning_hidden_dim

        self.input_norm = nn.LayerNorm(d_model)
        self.input_projection = nn.Linear(d_model, hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, d_model)
        self.activation = nn.SiLU()

        if condition_on_noise:
            self.noise_mlp: nn.Module | None = nn.Sequential(
                nn.Linear(1, conditioning_hidden_dim),
                nn.SiLU(),
                nn.Linear(conditioning_hidden_dim, hidden_dim),
            )
        else:
            self.noise_mlp = None

        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _prepare_noise_level(self, x: Tensor, noise_level: Tensor | float | None) -> Tensor:
        leading_shape = x.shape[:-1]
        if noise_level is None:
            return x.new_zeros((*leading_shape, 1))
        if isinstance(noise_level, (float, int)):
            return x.new_full((*leading_shape, 1), float(noise_level))

        assert noise_level.is_floating_point(), "noise_level must be floating point."
        assert noise_level.device == x.device, (
            f"noise_level is on {noise_level.device}, but x is on {x.device}."
        )
        assert noise_level.dtype == x.dtype, (
            f"noise_level has dtype {noise_level.dtype}, but x has dtype {x.dtype}."
        )
        if noise_level.ndim > 0 and noise_level.shape[-1] == 1:
            noise_level = noise_level.squeeze(-1)
        try:
            broadcast_level = torch.broadcast_to(noise_level, leading_shape)
        except RuntimeError as error:
            raise AssertionError(
                f"noise_level shape {noise_level.shape} cannot broadcast to {leading_shape}."
            ) from error
        return broadcast_level.unsqueeze(-1)

    def forward(self, x: Tensor, noise_level: Tensor | float | None = None) -> Tensor:
        """Denoise ``x`` while preserving all dimensions before ``d_model``."""
        assert x.ndim >= 2, f"Expected [..., d_model], got shape {x.shape}."
        assert x.shape[-1] == self.d_model
        assert x.is_floating_point()

        hidden = self.activation(self.input_projection(self.input_norm(x)))
        if self.noise_mlp is not None:
            conditioning = self._prepare_noise_level(x, noise_level)
            hidden = hidden + self.noise_mlp(conditioning)
        hidden = self.activation(self.hidden_projection(hidden))
        return x + self.output_projection(hidden)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


@torch.inference_mode()
def sanity_check_identity(
    d_model: int = 768,
    hidden_dim: int = 128,
    condition_on_noise: bool = True,
) -> None:
    """Verify exact identity behavior immediately after initialization."""
    denoiser = ResidualDenoiser(
        d_model=d_model,
        hidden_dim=hidden_dim,
        condition_on_noise=condition_on_noise,
    ).eval()
    x = torch.randn(2, 3, d_model)
    noise_level = torch.rand(2, 3) if condition_on_noise else None
    output = denoiser(x, noise_level=noise_level)
    torch.testing.assert_close(output, x, rtol=0.0, atol=0.0)
