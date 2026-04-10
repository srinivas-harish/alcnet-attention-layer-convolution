"""Attention utility tests."""

import torch

from src.main import _mask_diag, _mask_far, _nan_to_num, attn_stats_21, build_attn_tensor


class TestMaskFar:
    def test_shape(self):
        m = _mask_far(10, torch.device("cpu"), k=5)
        assert m.shape == (10, 10)

    def test_diagonal_is_false(self):
        m = _mask_far(10, torch.device("cpu"), k=5)
        # Diagonal distance is 0, which is < k=5, so should be False
        for i in range(10):
            assert not m[i, i].item()

    def test_far_elements_are_true(self):
        m = _mask_far(10, torch.device("cpu"), k=3)
        # Element (0, 5) has distance 5 >= 3, should be True
        assert m[0, 5].item()
        # Element (0, 2) has distance 2 < 3, should be False
        assert not m[0, 2].item()

    def test_symmetry(self):
        m = _mask_far(8, torch.device("cpu"), k=4)
        assert (m == m.T).all()

    def test_k_equals_1(self):
        m = _mask_far(5, torch.device("cpu"), k=1)
        # Only diagonal should be False
        assert not m[0, 0].item()
        assert m[0, 1].item()


class TestMaskDiag:
    def test_shape(self):
        m = _mask_diag(10, torch.device("cpu"))
        assert m.shape == (10, 10)

    def test_is_identity(self):
        m = _mask_diag(5, torch.device("cpu"))
        expected = torch.eye(5).bool()
        assert (m == expected).all()


class TestNanToNum:
    def test_replaces_nan(self):
        x = torch.tensor([1.0, float("nan"), 3.0])
        result = _nan_to_num(x)
        assert torch.isfinite(result).all()
        assert result[1].item() == 0.0

    def test_replaces_inf(self):
        x = torch.tensor([1.0, float("inf"), float("-inf")])
        result = _nan_to_num(x)
        assert torch.isfinite(result).all()

    def test_preserves_normal_values(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = _nan_to_num(x)
        assert torch.allclose(x, result)


class TestAttnStats21:
    def test_output_shape(self, fake_attention, batch_size):
        layers = [0, 1, 2]
        result = attn_stats_21(fake_attention, layers)
        assert result.shape == (batch_size, 21)

    def test_output_is_finite(self, fake_attention):
        layers = [0, 1, 2]
        result = attn_stats_21(fake_attention, layers)
        assert torch.isfinite(result).all()

    def test_single_layer(self, fake_attention, batch_size):
        # With 1 layer: 6 per-layer features + 3 global = 9
        result = attn_stats_21(fake_attention, [0])
        assert result.shape == (batch_size, 9)

    def test_two_layers(self, fake_attention, batch_size):
        # With 2 layers: 2*6 per-layer + 3 global = 15
        result = attn_stats_21(fake_attention, [0, 1])
        assert result.shape == (batch_size, 15)

    def test_uniform_attention_entropy(self, batch_size, n_heads, seq_len):
        """Uniform attention should have high entropy."""
        uniform = torch.ones(batch_size, n_heads, seq_len, seq_len) / seq_len
        result = attn_stats_21([uniform], [0])
        # Entropy feature is at index 2 (3rd feature)
        entropy = result[:, 2]
        # Entropy of uniform over seq_len tokens ≈ log(seq_len)
        import math
        expected_ent = math.log(seq_len)
        assert (entropy - expected_ent).abs().max() < 0.1

    def test_handles_zero_attention(self, batch_size, n_heads, seq_len):
        """Should handle near-zero attention without NaN."""
        zeros = torch.zeros(batch_size, n_heads, seq_len, seq_len) + 1e-15
        result = attn_stats_21([zeros], [0])
        assert torch.isfinite(result).all()


class TestBuildAttnTensor:
    def test_output_shape(self, fake_attention):
        layer_idx = [0, 1, 2]
        n_heads = fake_attention[0].shape[1]
        expected_channels = n_heads * len(layer_idx)
        resize_to = 32
        result = build_attn_tensor(fake_attention, layer_idx, resize_to)
        assert result.shape == (fake_attention[0].shape[0], expected_channels, resize_to, resize_to)

    def test_no_resize_needed(self, batch_size, n_heads):
        """When input size matches resize_to, no interpolation needed."""
        S = 64
        att = [torch.randn(batch_size, n_heads, S, S)]
        result = build_attn_tensor(att, [0], resize_to=S)
        assert result.shape[-1] == S

    def test_channel_normalization(self, batch_size, n_heads):
        S = 16
        att = [torch.randn(batch_size, n_heads, S, S) * 100 + 50]
        result = build_attn_tensor(att, [0], resize_to=S, channel_norm=True)
        # After channel norm, mean should be near 0 and std near 1
        for c in range(n_heads):
            channel = result[:, c]
            assert channel.mean().abs() < 0.5
            assert (channel.std() - 1.0).abs() < 0.5

    def test_no_channel_normalization(self, batch_size, n_heads):
        S = 16
        att = [torch.randn(batch_size, n_heads, S, S)]
        result_norm = build_attn_tensor(att, [0], resize_to=S, channel_norm=True)
        result_raw = build_attn_tensor(att, [0], resize_to=S, channel_norm=False)
        assert not torch.allclose(result_norm, result_raw)

    def test_handles_nan_in_input(self, batch_size, n_heads):
        S = 8
        att = [torch.randn(batch_size, n_heads, S, S)]
        att[0][0, 0, 0, 0] = float("nan")
        result = build_attn_tensor(att, [0], resize_to=S)
        assert torch.isfinite(result).all()
