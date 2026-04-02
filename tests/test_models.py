"""Tests for model components: ConvBlock, AttnCNN, MLP, GatedHybridClassifier."""

import pytest
import torch

from src.main import MLP, AttnCNN, ConvBlock, GatedHybridClassifier


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
