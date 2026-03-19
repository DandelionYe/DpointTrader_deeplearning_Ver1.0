# test_trainer_split_mode.py
"""
Phase 3: 回归测试 - Trainer split_mode 和 CV 切分策略

测试目标：
1. _make_eval_splits 在 use_embargo=True 时返回 embargo split，且 gap >= embargo_days
2. _make_eval_splits 在 use_nested_wf=True 时抛出 NotImplementedError
3. TrainResult holdout 字段在未启用 holdout 时为 None
"""
import pandas as pd
import numpy as np
import pytest
from trainer import _make_eval_splits, TrainResult


class TestMakeEvalSplits:
    """Test _make_eval_splits function."""

    def test_make_eval_splits_embargo_gap_is_correct(self):
        """
        use_embargo=True 时，验证 training 终点和 validation 起点之间的 gap >= embargo_days。
        
        这是真正的修复验证点：仅仅比较 validation set size 不够，
        必须直接验证 gap 是否存在。
        """
        n_samples = 200
        X = pd.DataFrame({
            "feat1": np.random.rand(n_samples),
            "feat2": np.random.rand(n_samples),
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        embargo_days = 5
        
        # 带 embargo 的 walkforward
        splits_with_embargo = _make_eval_splits(
            X, y, n_folds=4, train_start_ratio=0.5, wf_min_rows=20,
            use_embargo=True, embargo_days=embargo_days, use_nested_wf=False,
        )
        
        # 验证每个 fold 的 gap
        for i, ((X_tr, y_tr), (X_va, y_va)) in enumerate(splits_with_embargo):
            # 获取 training 集的最后一个索引和 validation 集的第一个索引
            train_end_idx = X_tr.index[-1]
            val_start_idx = X_va.index[0]
            
            # 计算 gap（索引差 - 1）
            # 例如：train_end=99, val_start=105，则 gap = 105 - 99 - 1 = 5
            gap = val_start_idx - train_end_idx - 1
            
            # 断言：gap >= embargo_days
            assert gap >= embargo_days, \
                f"Fold {i}: gap={gap} < embargo_days={embargo_days}. " \
                f"train_end_idx={train_end_idx}, val_start_idx={val_start_idx}"

    def test_make_eval_splits_no_embargo_gap_is_zero(self):
        """use_embargo=False 时，验证 training 终点和 validation 起点相邻（gap=0）。"""
        n_samples = 200
        X = pd.DataFrame({
            "feat1": np.random.rand(n_samples),
            "feat2": np.random.rand(n_samples),
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        # 标准 walkforward（无 embargo）
        splits_no_embargo = _make_eval_splits(
            X, y, n_folds=4, train_start_ratio=0.5, wf_min_rows=20,
            use_embargo=False, embargo_days=0, use_nested_wf=False,
        )
        
        # 验证每个 fold 的 gap
        for i, ((X_tr, y_tr), (X_va, y_va)) in enumerate(splits_no_embargo):
            train_end_idx = X_tr.index[-1]
            val_start_idx = X_va.index[0]
            
            # 无 embargo 时，gap 应该为 0（相邻）
            gap = val_start_idx - train_end_idx - 1
            
            assert gap == 0, \
                f"Fold {i}: gap should be 0 (no embargo), got {gap}. " \
                f"train_end_idx={train_end_idx}, val_start_idx={val_start_idx}"

    def test_make_eval_splits_raises_on_use_nested_wf_true(self):
        """use_nested_wf=True 时应抛出 NotImplementedError。"""
        n_samples = 100
        X = pd.DataFrame({"feat": np.random.rand(n_samples)})
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        with pytest.raises(NotImplementedError) as exc_info:
            _make_eval_splits(
                X, y, n_folds=3, train_start_ratio=0.5, wf_min_rows=20,
                use_embargo=False, embargo_days=0, use_nested_wf=True,
            )
        
        assert "use_nested_wf=True is declared but not integrated" in str(exc_info.value)
        assert "Do not silently ignore it" in str(exc_info.value)

    def test_make_eval_splits_default_behavior(self):
        """默认参数（use_embargo=False, use_nested_wf=False）返回标准 split。"""
        n_samples = 200
        X = pd.DataFrame({"feat": np.random.rand(n_samples)})
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        splits = _make_eval_splits(
            X, y, n_folds=4, train_start_ratio=0.5, wf_min_rows=20,
        )
        
        assert len(splits) == 4, f"Expected 4 splits, got {len(splits)}"
        
        # 验证训练集累积扩展
        prev_train_len = 0
        for (X_tr, y_tr), (X_va, y_va) in splits:
            assert len(X_tr) >= prev_train_len, "Training set should expand over folds"
            assert len(X_va) >= 20, "Validation set should meet min_rows requirement"
            prev_train_len = len(X_tr)


class TestTrainResultHoldoutFields:
    """Test TrainResult holdout fields are None when holdout is disabled."""

    def test_train_result_holdout_fields_are_none_by_default(self):
        """TrainResult 默认 holdout_metric/holdout_equity 为 None。"""
        result = TrainResult(
            best_config={},
            best_val_metric=0.1,
            best_val_final_equity_proxy=100000.0,
            search_log=pd.DataFrame(),
            feature_meta={},
            training_notes=[],
            global_best_updated=False,
            global_best_metric_prev=0.05,
            global_best_metric_new=0.1,
            candidate_best_metric=0.08,
            epsilon=0.01,
            not_updated_reason="test",
            best_so_far_path="/tmp/best.json",
            best_pool_path="/tmp/pool.json",
        )
        
        assert result.holdout_metric is None, "holdout_metric should be None by default"
        assert result.holdout_equity is None, "holdout_equity should be None by default"

    def test_train_result_holdout_fields_can_be_set(self):
        """TrainResult 可以显式设置 holdout_metric/holdout_equity。"""
        result = TrainResult(
            best_config={},
            best_val_metric=0.1,
            best_val_final_equity_proxy=100000.0,
            search_log=pd.DataFrame(),
            feature_meta={},
            training_notes=[],
            global_best_updated=False,
            global_best_metric_prev=0.05,
            global_best_metric_new=0.1,
            candidate_best_metric=0.08,
            epsilon=0.01,
            not_updated_reason="test",
            best_so_far_path="/tmp/best.json",
            best_pool_path="/tmp/pool.json",
            holdout_metric=0.12,
            holdout_equity=120000.0,
        )
        
        assert result.holdout_metric == 0.12, "holdout_metric should be settable"
        assert result.holdout_equity == 120000.0, "holdout_equity should be settable"
