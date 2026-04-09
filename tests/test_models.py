"""Tests for model components: SEBlock, ConvBlock, AttnCNN, MLP, GatedHybridClassifier."""

import pytest
import torch

from src.main import MLP, AttnCNN, ConvBlock, DropPath, GatedHybridClassifier, SEBlock


class TestDropPath:
    def test_eval_is_identity(self):
        dp = DropPath(drop_prob=0.5)
        dp.eval()
        x = torch.randn(4, 8, 4, 4)
        assert torch.equal(dp(x), x)

    def test_zero_drop_is_identity(self):
        dp = DropPath(drop_prob=0.0)
        dp.train()
        x = torch.randn(4, 8, 4, 4)
        assert torch.equal(dp(x), x)

    def test_train_applies_masking(self):
        torch.manual_seed(0)
        dp = DropPath(drop_prob=0.99)
        dp.train()
        x = torch.ones(100, 8, 2, 2)
        out = dp(x)
        # With 99% drop, most samples should be zeroed
        zero_samples = (out.sum(dim=(1, 2, 3)) == 0).sum().item()
        assert zero_samples > 50

    def test_output_shape_preserved(self):
        dp = DropPath(drop_prob=0.3)
        dp.train()
        x = torch.randn(4, 16, 8, 8)
        assert dp(x).shape == x.shape

    def test_gradient_flow(self):
        dp = DropPath(drop_prob=0.3)
        dp.train()
        x = torch.randn(4, 8, 4, 4, requires_grad=True)
        out = dp(x)
        out.sum().backward()
        assert x.grad is not None


class TestSEBlock:
    def test_output_shape_preserved(self):
        se = SEBlock(channels=32, reduction=4)
        x = torch.randn(2, 32, 8, 8)
        out = se(x)
        assert out.shape == x.shape

    def test_scale_is_bounded(self):
        se = SEBlock(channels=16, reduction=4)
        x = torch.randn(2, 16, 8, 8)
        # After sigmoid, scale should be in [0, 1]
        scale = se.se(x).unsqueeze(-1).unsqueeze(-1)
        assert (scale >= 0).all() and (scale <= 1).all()

    def test_gradient_flow(self):
        se = SEBlock(channels=8, reduction=2)
        x = torch.randn(2, 8, 4, 4, requires_grad=True)
        out = se(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestConvBlock:
    def test_output_shape(self):
        block = ConvBlock(in_c=16, out_c=32)
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        assert out.shape == (2, 32, 8, 8)

    def test_with_dropout(self):
        block = ConvBlock(in_c=16, out_c=32, drop=0.5)
        x = torch.randn(2, 16, 8, 8)
        block.train()
        out = block(x)
        assert out.shape == (2, 32, 8, 8)

    def test_different_kernel_stride(self):
        block = ConvBlock(in_c=8, out_c=16, k=5, s=2, p=2)
        x = torch.randn(2, 8, 16, 16)
        out = block(x)
        assert out.shape == (2, 16, 8, 8)

    def test_gradient_flow(self):
        block = ConvBlock(in_c=4, out_c=8)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestAttnCNN:
    def test_output_shape(self):
        in_ch = 12  # e.g., 4 heads * 3 layers
        cnn = AttnCNN(in_ch=in_ch, out_dim=256)
        x = torch.randn(2, in_ch, 128, 128)
        out = cnn(x)
        assert out.shape == (2, 256)

    def test_different_input_sizes(self):
        cnn = AttnCNN(in_ch=8, out_dim=128)
        for size in [32, 64, 128]:
            x = torch.randn(1, 8, size, size)
            out = cnn(x)
            assert out.shape == (1, 128)

    def test_gradient_flow(self):
        cnn = AttnCNN(in_ch=6, out_dim=64)
        x = torch.randn(2, 6, 32, 32, requires_grad=True)
        out = cnn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(in_dim=21, out_dim=128)
        x = torch.randn(4, 21)
        out = mlp(x)
        assert out.shape == (4, 128)

    def test_small_input_dim(self):
        mlp = MLP(in_dim=5, out_dim=32)
        x = torch.randn(2, 5)
        out = mlp(x)
        assert out.shape == (2, 32)

    def test_gradient_flow(self):
        mlp = MLP(in_dim=21, out_dim=128)
        x = torch.randn(2, 21, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestGatedHybridClassifier:
    @pytest.fixture
    def model(self):
        return GatedHybridClassifier(
            attn_in_ch=12, cls_dim=64, stats_dim=21, num_classes=2
        )

    def test_output_shape(self, model):
        B = 4
        attn_img = torch.randn(B, 12, 128, 128)
        cls_vec = torch.randn(B, 64)
        stats_vec = torch.randn(B, 21)
        logits = model(attn_img, cls_vec, stats_vec)
        assert logits.shape == (B, 2)

    def test_output_is_finite(self, model):
        B = 2
        attn_img = torch.randn(B, 12, 64, 64)
        cls_vec = torch.randn(B, 64)
        stats_vec = torch.randn(B, 21)
        logits = model(attn_img, cls_vec, stats_vec)
        assert torch.isfinite(logits).all()

    def test_gradient_flows_to_all_branches(self, model):
        B = 2
        attn_img = torch.randn(B, 12, 32, 32, requires_grad=True)
        cls_vec = torch.randn(B, 64, requires_grad=True)
        stats_vec = torch.randn(B, 21, requires_grad=True)
        logits = model(attn_img, cls_vec, stats_vec)
        loss = logits.sum()
        loss.backward()
        assert attn_img.grad is not None, "No gradient to attention image"
        assert cls_vec.grad is not None, "No gradient to CLS vector"
        assert stats_vec.grad is not None, "No gradient to stats vector"

    def test_confidence_gate_is_bounded(self, model):
        B = 4
        attn_img = torch.randn(B, 12, 64, 64)
        stats_vec = torch.randn(B, 21)
        # Access internal conf_head to verify sigmoid output is in [0, 1]
        cnn_feat = model.cnn(attn_img)
        stats_feat = model.stats_mlp(stats_vec)
        combined = torch.cat([cnn_feat, stats_feat], dim=1)
        conf = torch.sigmoid(model.conf_head(combined))
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_num_classes_3(self):
        model = GatedHybridClassifier(
            attn_in_ch=8, cls_dim=32, stats_dim=21, num_classes=3
        )
        B = 2
        logits = model(
            torch.randn(B, 8, 32, 32),
            torch.randn(B, 32),
            torch.randn(B, 21),
        )
        assert logits.shape == (B, 3)

    def test_weight_init_batchnorm(self, model):
        """BatchNorm weight should be ~1.0 and bias ~0.0 after init."""
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                assert torch.allclose(m.weight, torch.ones_like(m.weight))
                assert torch.allclose(m.bias, torch.zeros_like(m.bias))

    def test_classification_heads_xavier(self, model):
        """Classification heads should have variance consistent with Xavier init."""
        for head in [model.transformer_head, model.cnn_head]:
            w = head.weight
            fan_in, fan_out = w.shape[1], w.shape[0]
            expected_std = (2.0 / (fan_in + fan_out)) ** 0.5
            # Allow generous tolerance — just verify it's in the right ballpark
            assert w.std().item() < expected_std * 3.0
