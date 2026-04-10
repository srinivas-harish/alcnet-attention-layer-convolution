"""Config and runner tests."""


from src.main import AlcnetCfg, effective_num_workers, parse_layers, pick_device


class TestPickDevice:
    def test_force_cpu(self):
        d = pick_device("cpu")
        assert d.type == "cpu"

    def test_force_none_returns_device(self):
        d = pick_device(None)
        assert d.type in ("cpu", "cuda")

    def test_default_is_auto(self):
        d = pick_device()
        assert d.type in ("cpu", "cuda")


class TestEffectiveNumWorkers:
    def test_none_returns_zero(self):
        assert effective_num_workers(None) == 0

    def test_positive_value(self, monkeypatch):
        monkeypatch.delenv("IN_CELERY", raising=False)
        assert effective_num_workers(4) == 4

    def test_negative_clamped_to_zero(self, monkeypatch):
        monkeypatch.delenv("IN_CELERY", raising=False)
        assert effective_num_workers(-1) == 0

    def test_celery_env_forces_zero(self, monkeypatch):
        monkeypatch.setenv("IN_CELERY", "1")
        assert effective_num_workers(4) == 0

    def test_celery_env_not_set(self, monkeypatch):
        monkeypatch.setenv("IN_CELERY", "0")
        assert effective_num_workers(4) == 4


class TestParseLayers:
    def test_positive_indices(self):
        result = parse_layers((0, 1, 2), n_layers=6)
        assert result == [0, 1, 2]

    def test_negative_indices(self):
        result = parse_layers((-3, -2, -1), n_layers=6)
        assert result == [3, 4, 5]

    def test_mixed_indices(self):
        result = parse_layers((0, -1), n_layers=4)
        assert result == [0, 3]

    def test_out_of_range_excluded(self):
        result = parse_layers((0, 10, -1), n_layers=4)
        assert result == [0, 3]

    def test_deduplication(self):
        result = parse_layers((0, 0, -4), n_layers=4)
        assert result == [0]

    def test_empty_input(self):
        result = parse_layers((), n_layers=4)
        assert result == []

    def test_sorted_output(self):
        result = parse_layers((-1, 0, -2), n_layers=6)
        assert result == sorted(result)


class TestAlcnetCfg:
    def test_defaults(self):
        cfg = AlcnetCfg()
        assert cfg.model_name == "roberta-large"
        assert cfg.task_name == "rte"
        assert cfg.max_len == 128
        assert cfg.batch_size == 8
        assert cfg.epochs == 3
        assert cfg.seed == 42
        assert cfg.attn_layers == (-3, -2, -1)

    def test_custom_values(self):
        cfg = AlcnetCfg(epochs=10, batch_size=16, lr_head=1e-3)
        assert cfg.epochs == 10
        assert cfg.batch_size == 16
        assert cfg.lr_head == 1e-3

    def test_optional_caps_default_none(self):
        cfg = AlcnetCfg()
        assert cfg.max_train_samples is None
        assert cfg.max_val_samples is None
        assert cfg.max_train_batches is None
        assert cfg.max_val_batches is None


class TestRunnerOverrideValidation:
    def test_allowed_keys_exist(self):
        from src.runner import _ALLOWED_OVERRIDE_KEYS
        # Should contain all dataclass fields
        cfg = AlcnetCfg()
        for field_name in cfg.__dataclass_fields__:
            assert field_name in _ALLOWED_OVERRIDE_KEYS

    def test_override_rejects_unknown_keys(self):
        from src.runner import _ALLOWED_OVERRIDE_KEYS
        assert "malicious_field" not in _ALLOWED_OVERRIDE_KEYS
        assert "__class__" not in _ALLOWED_OVERRIDE_KEYS
