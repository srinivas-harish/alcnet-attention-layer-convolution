"""Shared fixtures."""

import pytest
import torch


@pytest.fixture
def device():
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
    att_list = []
    for _ in range(n_layers):
        raw = torch.randn(batch_size, n_heads, seq_len, seq_len)
        att = torch.softmax(raw, dim=-1)
        att_list.append(att)
    return att_list
