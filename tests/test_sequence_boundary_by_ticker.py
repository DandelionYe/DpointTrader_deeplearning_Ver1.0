"""
测试序列边界（按ticker隔离）
"""
import numpy as np
import pandas as pd
import pytest

from sequence_builder import build_panel_sequences


@pytest.fixture
def sample_panel():
    """创建样本panel数据"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    tickers = ["A", "B", "C"]

    rows = []
    for ticker in tickers:
        for date in dates:
            rows.append({
                "date": date,
                "ticker": ticker,
                "feature1": np.random.randn(),
                "feature2": np.random.randn(),
            })

    df = pd.DataFrame(rows)
    df["label"] = (df["feature1"] > 0).astype(int)
    return df


class TestSequenceBoundaryByTicker:
    """测试序列边界按ticker隔离"""

    def test_window_contains_single_ticker_only(self, sample_panel):
        """测试窗口内只包含同一个ticker"""
        y = pd.Series(np.random.randn(len(sample_panel)), index=sample_panel.index)

        seq_len = 20
        bundle = build_panel_sequences(
            sample_panel[["date", "ticker", "feature1", "feature2"]],
            y,
            date_col="date",
            ticker_col="ticker",
            seq_len=seq_len,
        )

        # 验证每个窗口的source_index对应的ticker一致
        for idx in range(len(bundle.meta_df)):
            source_idx = bundle.meta_df.iloc[idx]["source_index"]
            window_ticker = bundle.meta_df.iloc[idx]["ticker"]

            # source_index的ticker应该等于meta_df的ticker
            assert sample_panel.iloc[source_idx]["ticker"] == window_ticker

    def test_no_cross_ticker_in_window(self, sample_panel):
        """测试不允许跨ticker拼接窗口"""
        y = pd.Series(np.random.randn(len(sample_panel)), index=sample_panel.index)

        seq_len = 20
        bundle = build_panel_sequences(
            sample_panel[["date", "ticker", "feature1", "feature2"]],
            y,
            date_col="date",
            ticker_col="ticker",
            seq_len=seq_len,
        )

        # 按ticker分组检查
        for ticker in sample_panel["ticker"].unique():
            ticker_mask = bundle.meta_df["ticker"] == ticker
            ticker_sequences = bundle.X_seq[ticker_mask]

            # 确保序列数量正确
            expected_count = len(sample_panel[sample_panel["ticker"] == ticker]) - seq_len + 1
            assert len(ticker_sequences) == expected_count


class TestSequencePredictionShape:
    """测试序列预测形状"""

    def test_prediction_rows_less_than_original(self, sample_panel):
        """测试预测结果行数少于原始样本数"""
        y = pd.Series(np.random.randn(len(sample_panel)), index=sample_panel.index)

        seq_len = 20
        bundle = build_panel_sequences(
            sample_panel[["date", "ticker", "feature1", "feature2"]],
            y,
            date_col="date",
            ticker_col="ticker",
            seq_len=seq_len,
        )

        # 预测结果行数 = 原样本数 - 每个ticker的warmup缺口
        # 每个ticker损失 seq_len - 1 行
        n_tickers = sample_panel["ticker"].nunique()
        expected_rows = len(sample_panel) - n_tickers * (seq_len - 1)

        assert len(bundle.X_seq) == expected_rows

    def test_return_columns_complete(self, sample_panel):
        """测试返回列完整"""
        y = pd.Series(np.random.randn(len(sample_panel)), index=sample_panel.index)

        seq_len = 20
        bundle = build_panel_sequences(
            sample_panel[["date", "ticker", "feature1", "feature2"]],
            y,
            date_col="date",
            ticker_col="ticker",
            seq_len=seq_len,
        )

        # 检查meta_df列
        assert "source_index" in bundle.meta_df.columns
        assert "date" in bundle.meta_df.columns
        assert "ticker" in bundle.meta_df.columns

        # 检查X_seq形状
        assert bundle.X_seq.ndim == 3  # (n_samples, seq_len, n_features)
        assert bundle.X_seq.shape[1] == seq_len
        assert bundle.X_seq.shape[2] == 2  # feature1, feature2
