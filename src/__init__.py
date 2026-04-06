"""ALCNet: Attention-Layer Convolutional Network with FiLM Conditioning."""

__all__ = [
    "AlcnetCfg",
    "GatedHybridClassifier",
    "train_and_eval",
]

from .main import AlcnetCfg, GatedHybridClassifier, train_and_eval

