"""Shared test fixtures for ALCNet tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Use CPU for all tests (no GPU required)."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def n_heads():
    return 4


@pytest.fixture
def n_layers():
    return 3


@pytest.fixture
def fake_attention(batch_size, n_heads, seq_len, n_layers):
    """Generate fake attention tensors simulating transformer output.

    Returns a list of (B, H, S, S) tensors, one per layer.
    Values are softmax-normalized along the last dim to mimic real attention.
    """
    att_list = []
    for _ in range(n_layers):
        raw = torch.randn(batch_size, n_heads, seq_len, seq_len)
        att = torch.softmax(raw, dim=-1)
        att_list.append(att)
    return att_list
