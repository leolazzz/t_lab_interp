import unittest

import torch

from src.denoiser import ResidualDenoiser, sanity_check_identity


class DenoiserTests(unittest.TestCase):
    def test_initial_model_is_exact_identity_and_preserves_shape(self) -> None:
        model = ResidualDenoiser(
            d_model=6,
            hidden_dim=10,
            condition_on_noise=True,
            conditioning_hidden_dim=4,
        )
        x = torch.randn(2, 3, 6)
        output = model(x, noise_level=torch.rand(2, 3))
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(output, x, atol=0.0, rtol=0.0)
        self.assertGreater(model.count_parameters(), 0)

    def test_module_sanity_check(self) -> None:
        sanity_check_identity(d_model=8, hidden_dim=4)


if __name__ == "__main__":
    unittest.main()
