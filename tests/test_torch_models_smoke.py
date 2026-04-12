"""
测试Torch模型smoke测试
"""
import os

import numpy as np
import pandas as pd
import pytest

from models import TORCH_AVAILABLE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTorchModelsSmoke:
    """测试Torch模型基本路径"""

    @pytest.fixture
    def tiny_tabular_data(self):
        """创建小型tabular数据"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y = pd.Series((np.random.randn(n_samples) > 0).astype(float))

        return X, y

    @pytest.fixture
    def tiny_sequence_data(self):
        """创建小型序列数据"""
        np.random.seed(42)
        n_samples = 50
        seq_len = 10
        n_features = 5

        # 创建序列数据
        X_seq = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        y_seq = (np.random.randn(n_samples) > 0).astype(np.float32)

        return X_seq, y_seq, seq_len

    @pytest.fixture
    def local_tmpdir(self):
        path = os.path.join(".local", "tmp", "torch_smoke_artifacts")
        os.makedirs(path, exist_ok=True)
        yield path

    def test_mlp_train_predict_smoke(self, tiny_tabular_data):
        """测试MLP训练+预测路径"""
        from models import predict_pytorch_model_tabular, resolve_torch_device, train_pytorch_model

        X, y = tiny_tabular_data
        device = resolve_torch_device("cpu")

        config = {
            "model_type": "mlp",
            "hidden_dims": [32, 16],
            "dropout_rate": 0.1,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "epochs": 2,
            "batch_size": 32,
        }

        # 训练
        model = train_pytorch_model(X, y, config, device=device)

        # 预测
        scores = predict_pytorch_model_tabular(model, X, device)

        assert len(scores) == len(X)
        assert not scores.isna().any()

    def test_lstm_train_predict_smoke(self, tiny_tabular_data):
        """测试LSTM训练+预测路径"""
        from models import predict_pytorch_model, resolve_torch_device, train_pytorch_model

        X, y = tiny_tabular_data
        device = resolve_torch_device("cpu")

        config = {
            "model_type": "lstm",
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout_rate": 0.1,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "epochs": 2,
            "batch_size": 32,
            "seq_len": 10,
            "bidirectional": False,
        }

        # 训练
        model = train_pytorch_model(X, y, config, device=device)

        # 预测（使用向后兼容的predict_pytorch_model）
        scores = predict_pytorch_model(model, X, device, seq_len=10)

        # 序列模型预测会少seq_len-1行
        assert len(scores) == len(X) - 10 + 1
        assert not scores.isna().any()

    def test_gru_train_predict_smoke(self, tiny_tabular_data):
        """测试GRU训练+预测路径"""
        from models import predict_pytorch_model, resolve_torch_device, train_pytorch_model

        X, y = tiny_tabular_data
        device = resolve_torch_device("cpu")

        config = {
            "model_type": "gru",
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout_rate": 0.1,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "epochs": 2,
            "batch_size": 32,
            "seq_len": 10,
            "bidirectional": False,
        }

        # 训练
        model = train_pytorch_model(X, y, config, device=device)

        # 预测
        scores = predict_pytorch_model(model, X, device, seq_len=10)

        assert len(scores) == len(X) - 10 + 1
        assert not scores.isna().any()

    def test_cnn_train_predict_smoke(self, tiny_tabular_data):
        """测试CNN训练+预测路径"""
        from models import predict_pytorch_model, resolve_torch_device, train_pytorch_model

        X, y = tiny_tabular_data
        device = resolve_torch_device("cpu")

        config = {
            "model_type": "cnn",
            "num_filters": 32,
            "kernel_sizes": [3, 5],
            "dropout_rate": 0.1,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "epochs": 2,
            "batch_size": 32,
            "seq_len": 10,
        }

        # 训练
        model = train_pytorch_model(X, y, config, device=device)

        # 预测
        scores = predict_pytorch_model(model, X, device, seq_len=10)

        assert len(scores) == len(X) - 10 + 1
        assert not scores.isna().any()

    def test_transformer_train_predict_smoke(self, tiny_tabular_data):
        """测试Transformer训练+预测路径"""
        from models import predict_pytorch_model, resolve_torch_device, train_pytorch_model

        X, y = tiny_tabular_data
        device = resolve_torch_device("cpu")

        config = {
            "model_type": "transformer",
            "d_model": 32,
            "nhead": 4,
            "dim_feedforward": 64,
            "num_layers": 1,
            "dropout_rate": 0.1,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "epochs": 2,
            "batch_size": 32,
            "seq_len": 10,
        }

        # 训练
        model = train_pytorch_model(X, y, config, device=device)

        # 预测
        scores = predict_pytorch_model(model, X, device, seq_len=10)

        assert len(scores) == len(X) - 10 + 1
        assert not scores.isna().any()

    def test_auto_batch_tune_rolls_back_from_overshoot(self, monkeypatch):
        import models

        class FakeDevice:
            type = "cuda"

        observed_batches = []

        def fake_probe(**kwargs):
            batch_size = kwargs["batch_size"]
            observed_batches.append(batch_size)
            if batch_size <= 736:
                util_map = {
                    64: 0.08,
                    128: 0.16,
                    256: 0.30,
                    512: 0.58,
                    736: 0.82,
                }
                util = util_map[batch_size]
            elif batch_size == 832:
                util = 0.90
            else:
                util = 1.03
            allocated = int(util * 8 * (1024 ** 3))
            reserved = allocated + 128 * (1024 ** 2)
            return util, allocated, reserved

        monkeypatch.setattr(models, "_probe_sequence_batch_memory", fake_probe)
        monkeypatch.setattr(models, "_cuda_total_memory", lambda device: 8 * (1024 ** 3))

        selected = models._auto_tune_sequence_batch_size(
            mode="train",
            model_type="lstm",
            input_dim=16,
            config={"auto_batch_tune": True, "target_vram_util": 0.88},
            X=np.zeros((4000, 10, 16), dtype=np.float32),
            y=np.zeros(4000, dtype=np.float32),
            device=FakeDevice(),
            base_batch_size=64,
            loader_settings={},
        )

        assert selected == 832
        assert 928 in observed_batches

    def test_sequence_prediction_output_is_float32(self):
        import torch

        from models import predict_pytorch_model_sequence, resolve_torch_device

        class HalfPrecisionSequenceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._predict_batch_size = 4
                self._trained_batch_size = 4
                self._loader_settings = {}
                self._auto_batch_tune = False
                self._use_amp = False
                self._use_tf32 = False

            def forward(self, x):
                logits = x.mean(dim=(1, 2), keepdim=True)
                return logits.to(torch.float16)

        model = HalfPrecisionSequenceModel()
        device = resolve_torch_device("cpu")
        X_seq = np.random.randn(6, 5, 3).astype(np.float32)
        meta_df = pd.DataFrame(
            {
                "source_index": np.arange(6),
                "date": pd.date_range("2020-01-01", periods=6, freq="B"),
                "ticker": ["A"] * 6,
            }
        )

        scores = predict_pytorch_model_sequence(model, X_seq, meta_df, device)

        assert str(scores.dtype) == "float32"
        assert not scores.isna().any()

    def test_auto_batch_tune_reuses_cache(self, monkeypatch):
        import models

        class FakeDevice:
            type = "cuda"

            def __str__(self):
                return "cuda"

        models._AUTO_BATCH_TUNE_CACHE.clear()
        probe_calls = {"count": 0}

        def fake_probe(**kwargs):
            probe_calls["count"] += 1
            batch_size = kwargs["batch_size"]
            util = 0.89 if batch_size >= 832 else 0.40
            allocated = int(util * 8 * (1024 ** 3))
            reserved = allocated
            return util, allocated, reserved

        monkeypatch.setattr(models, "_probe_sequence_batch_memory", fake_probe)
        monkeypatch.setattr(models, "_cuda_total_memory", lambda device: 8 * (1024 ** 3))

        common_kwargs = dict(
            mode="train",
            model_type="lstm",
            input_dim=16,
            config={"auto_batch_tune": True, "target_vram_util": 0.88, "seq_len": 10, "hidden_dim": 32, "num_layers": 1},
            X=np.zeros((4000, 10, 16), dtype=np.float32),
            y=np.zeros(4000, dtype=np.float32),
            device=FakeDevice(),
            base_batch_size=832,
            loader_settings={},
        )

        selected_1 = models._auto_tune_sequence_batch_size(**common_kwargs)
        selected_2 = models._auto_tune_sequence_batch_size(**common_kwargs)

        assert selected_1 == 832
        assert selected_2 == 832
        assert probe_calls["count"] == 1

    def test_cuda_oom_detection_accepts_runtime_error_variants(self):
        import models

        assert models._is_cuda_oom_error(RuntimeError("CUDA out of memory")) is True
        assert models._is_cuda_oom_error(RuntimeError("CUDA error: out of memory")) is True
        assert models._is_cuda_oom_error(RuntimeError("cudaErrorMemoryAllocation")) is True
        assert models._is_cuda_oom_error(RuntimeError("unrelated runtime error")) is False

    def test_torch_models_save_as_state_dict_artifact(self, tiny_tabular_data, local_tmpdir):
        from models import load_saved_model, save_trained_model
        from panel_trainer import train_panel_model

        X, y = tiny_tabular_data
        panel_X = X.copy()
        panel_X["date"] = pd.date_range("2020-01-01", periods=len(X), freq="B")
        panel_X["ticker"] = ["A"] * len(X)
        config = {
            "model_type": "mlp",
            "device": "cpu",
            "model_params": {
                "hidden_dims": [16, 8],
                "hidden_dim": 16,
                "dropout_rate": 0.1,
                "learning_rate": 0.01,
                "weight_decay": 1e-4,
                "epochs": 1,
                "batch_size": 16,
            },
        }

        model, _ = train_panel_model(panel_X, y, config, date_col="date", ticker_col="ticker", seed=42)

        saved_path = save_trained_model(model, config, os.path.join(local_tmpdir, "torch_model"))
        assert os.path.isdir(saved_path)
        assert os.path.exists(os.path.join(saved_path, "model_state.pt"))
        assert os.path.exists(os.path.join(saved_path, "model_config.json"))
        assert os.path.exists(os.path.join(saved_path, "feature_meta.json"))
        assert os.path.exists(os.path.join(saved_path, "normalizer.pkl"))

        restored = load_saved_model(saved_path)
        assert hasattr(restored, "state_dict")
        assert getattr(restored, "_preprocessor", None) is not None

    def test_predict_panel_marks_proba_unavailable_for_non_probabilistic_model(self):
        from panel_trainer import predict_panel

        class PredictOnlyModel:
            def predict(self, X):
                return np.ones(len(X), dtype=np.float32)

        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="B"),
                "ticker": ["A"] * 4,
                "feature_1": [0.1, 0.2, 0.3, 0.4],
            }
        )

        pred = predict_panel(PredictOnlyModel(), X, date_col="date", ticker_col="ticker")

        assert "proba" in pred.columns
        assert "probability_available" in pred.columns
        assert pred["probability_available"].eq(False).all()
        assert pred["proba"].isna().all()

    def test_predict_panel_exposes_probabilities_for_torch_binary_tabular_model(self, tiny_tabular_data):
        from panel_trainer import predict_panel, train_panel_model

        X, y = tiny_tabular_data
        panel_X = X.copy()
        panel_X["date"] = pd.date_range("2020-01-01", periods=len(X), freq="B")
        panel_X["ticker"] = ["A"] * len(X)
        config = {
            "model_type": "mlp",
            "device": "cpu",
            "model_params": {
                "hidden_dims": [16, 8],
                "hidden_dim": 16,
                "dropout_rate": 0.1,
                "learning_rate": 0.01,
                "weight_decay": 1e-4,
                "epochs": 1,
                "batch_size": 16,
            },
        }

        model, _ = train_panel_model(panel_X, y, config, date_col="date", ticker_col="ticker", seed=42)
        pred = predict_panel(model, panel_X, date_col="date", ticker_col="ticker")

        assert pred["probability_available"].eq(True).all()
        assert pred["proba"].between(0.0, 1.0).all()
