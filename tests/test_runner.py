"""Runner tests."""

from unittest.mock import patch

from src.main import AlcnetCfg
from src.runner import _ALLOWED_OVERRIDE_KEYS, run_from_req


class TestRunFromReq:
    @patch("src.runner.train_and_eval")
    def test_default_config(self, mock_train):
        mock_train.return_value = {"best": {"val_acc": 0.5}}
        run_from_req({}, run_id="test-run")
        cfg = mock_train.call_args.kwargs["cfg"]
        assert cfg.epochs == 3
        assert cfg.batch_size == 8
        assert cfg.max_len == 128

    @patch("src.runner.train_and_eval")
    def test_epochs_override(self, mock_train):
        mock_train.return_value = {"best": {"val_acc": 0.5}}
        run_from_req({"epochs": 10}, run_id="test-run")
        cfg = mock_train.call_args.kwargs["cfg"]
        assert cfg.epochs == 10

    @patch("src.runner.train_and_eval")
    def test_batch_size_override(self, mock_train):
        mock_train.return_value = {"best": {"val_acc": 0.5}}
        run_from_req({"batch_size": 32}, run_id="test-run")
        cfg = mock_train.call_args.kwargs["cfg"]
        assert cfg.batch_size == 32

    @patch("src.runner.train_and_eval")
    def test_ablation_override(self, mock_train):
        mock_train.return_value = {"best": {"val_acc": 0.5}}
        run_from_req({"ablation": "no_cnn"}, run_id="test-run")
        cfg = mock_train.call_args.kwargs["cfg"]
        assert cfg.ablation == "no_cnn"

    @patch("src.runner.train_and_eval")
    def test_overrides_dict(self, mock_train):
        mock_train.return_value = {"best": {"val_acc": 0.5}}
        run_from_req({"overrides": {"lr_head": 1e-3}}, run_id="test-run")
        cfg = mock_train.call_args.kwargs["cfg"]
        assert cfg.lr_head == 1e-3

    @patch("src.runner.train_and_eval")
    def test_unknown_override_ignored(self, mock_train):
        mock_train.return_value = {"best": {"val_acc": 0.5}}
        run_from_req({"overrides": {"unknown_key": 42}}, run_id="test-run")
        cfg = mock_train.call_args.kwargs["cfg"]
        assert not hasattr(cfg, "unknown_key")

    @patch("src.runner.train_and_eval")
    def test_gradient_checkpointing(self, mock_train):
        mock_train.return_value = {"best": {"val_acc": 0.5}}
        run_from_req({"gradient_checkpointing": False}, run_id="test-run")
        cfg = mock_train.call_args.kwargs["cfg"]
        assert cfg.gradient_checkpointing is False

    @patch("src.runner.train_and_eval")
    def test_device_passthrough(self, mock_train):
        mock_train.return_value = {"best": {"val_acc": 0.5}}
        run_from_req({"device": "cpu"}, run_id="test-run")
        device = mock_train.call_args.kwargs["device"]
        assert str(device) == "cpu"


class TestAllowedOverrideKeys:
    def test_allowed_keys_include_core_fields(self):
        for field in ["epochs", "batch_size", "lr_head", "lr_encoder", "weight_decay"]:
            assert field in _ALLOWED_OVERRIDE_KEYS

    def test_allowed_keys_match_config(self):
        from dataclasses import fields as dataclass_fields
        expected = frozenset(f.name for f in dataclass_fields(AlcnetCfg))
        assert expected == _ALLOWED_OVERRIDE_KEYS
