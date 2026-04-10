"""FiLM conditioning tests."""

import torch

from src.main import AttnCNN, FiLMGenerator, GatedHybridClassifier, apply_film


class TestFiLMGenerator:
    def test_output_shapes(self):
        gen = FiLMGenerator(cond_dim=21, n_channels=128)
        cond = torch.randn(4, 21)
        gamma, beta = gen(cond)
        assert gamma.shape == (4, 128)
        assert beta.shape == (4, 128)

    def test_initial_gamma_near_one(self):
        gen = FiLMGenerator(cond_dim=10, n_channels=64)
        cond = torch.zeros(2, 10)
        gamma, _beta = gen(cond)
        # With zero input and zero weights, gamma should be bias (init to 1.0)
        assert (gamma - 1.0).abs().max() < 0.01

    def test_initial_beta_near_zero(self):
        gen = FiLMGenerator(cond_dim=10, n_channels=64)
        cond = torch.zeros(2, 10)
        _gamma, beta = gen(cond)
        # Beta bias initialized to 0
        assert beta.abs().max() < 0.01

    def test_gradient_flow(self):
        gen = FiLMGenerator(cond_dim=21, n_channels=128)
        cond = torch.randn(2, 21, requires_grad=True)
        gamma, beta = gen(cond)
        loss = gamma.sum() + beta.sum()
        loss.backward()
        assert cond.grad is not None


class TestApplyFiLM:
    def test_4d_features(self):
        features = torch.randn(2, 8, 16, 16)
        gamma = torch.ones(2, 8)
        beta = torch.zeros(2, 8)
        out = apply_film(features, gamma, beta)
        assert torch.allclose(out, features)

    def test_4d_scaling(self):
        features = torch.ones(2, 4, 8, 8)
        gamma = torch.full((2, 4), 2.0)
        beta = torch.full((2, 4), 1.0)
        out = apply_film(features, gamma, beta)
        # 2 * 1 + 1 = 3
        assert torch.allclose(out, torch.full_like(out, 3.0))

    def test_2d_features(self):
        features = torch.randn(2, 8)
        gamma = torch.ones(2, 8) * 3.0
        beta = torch.zeros(2, 8)
        out = apply_film(features, gamma, beta)
        assert torch.allclose(out, features * 3.0)

    def test_per_sample_conditioning(self):
        """Different samples in batch get different conditioning."""
        features = torch.ones(2, 4, 8, 8)
        gamma = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
        beta = torch.zeros(2, 4)
        out = apply_film(features, gamma, beta)
        # Sample 0: 1*1 = 1, Sample 1: 2*1 = 2
        assert torch.allclose(out[0], torch.ones(4, 8, 8))
        assert torch.allclose(out[1], torch.full((4, 8, 8), 2.0))


class TestAttnCNNWithFiLM:
    def test_with_film_params(self):
        cnn = AttnCNN(in_ch=12, out_dim=256)
        x = torch.randn(2, 12, 64, 64)
        film_params = {
            "gamma1": torch.ones(2, 128),
            "beta1": torch.zeros(2, 128),
            "gamma2": torch.ones(2, 256),
            "beta2": torch.zeros(2, 256),
        }
        out = cnn(x, film_params=film_params)
        assert out.shape == (2, 256)

    def test_without_film_params(self):
        cnn = AttnCNN(in_ch=12, out_dim=256)
        x = torch.randn(2, 12, 64, 64)
        out = cnn(x, film_params=None)
        assert out.shape == (2, 256)

    def test_film_changes_output(self):
        """FiLM conditioning should change the CNN output."""
        cnn = AttnCNN(in_ch=8, out_dim=128)
        cnn.eval()
        x = torch.randn(2, 8, 32, 32)

        out_no_film = cnn(x, film_params=None)

        film_params = {
            "gamma1": torch.full((2, 128), 5.0),
            "beta1": torch.full((2, 128), 10.0),
            "gamma2": torch.full((2, 256), 5.0),
            "beta2": torch.full((2, 256), 10.0),
        }
        out_with_film = cnn(x, film_params=film_params)

        assert not torch.allclose(out_no_film, out_with_film, atol=1e-3)


class TestGatedHybridWithFiLM:
    def test_gradient_flows_through_film(self):
        """Verify that gradients from the loss reach the stats vector through FiLM."""
        model = GatedHybridClassifier(
            attn_in_ch=12, cls_dim=64, stats_dim=21, num_classes=2
        )
        attn_img = torch.randn(2, 12, 32, 32)
        cls_vec = torch.randn(2, 64)
        stats_vec = torch.randn(2, 21, requires_grad=True)

        logits = model(attn_img, cls_vec, stats_vec)
        loss = logits.sum()
        loss.backward()

        # Stats should get gradients through both the MLP path AND the FiLM path
        assert stats_vec.grad is not None
        assert stats_vec.grad.abs().sum() > 0

    def test_film_generators_have_parameters(self):
        model = GatedHybridClassifier(
            attn_in_ch=12, cls_dim=64, stats_dim=21, num_classes=2
        )
        film1_params = list(model.film1.parameters())
        film2_params = list(model.film2.parameters())
        assert len(film1_params) > 0
        assert len(film2_params) > 0
