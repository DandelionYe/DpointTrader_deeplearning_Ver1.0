"""
测试序列构建器
"""
import pytest
import pandas as pd
import numpy as np

from sequence_builder import build_panel_sequences


@pytest.fixture
def sample_panel():
    """创建样本panel数据"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    tickers = ["A", "B"]
    
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


class TestSequenceBuilder:
    """测试序列构建器"""
    
    def test_single_ticker_window_count(self):
        """测试单个ticker生成窗口数"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        
        rows = []
        for date in dates:
            rows.append({
                "date": date,
                "ticker": "A",
                "feature1": np.random.randn(),
            })
        
        df = pd.DataFrame(rows)
        y = pd.Series(np.random.randn(len(df)), index=df.index)
        
        seq_len = 10
        bundle = build_panel_sequences(
            df, y,
            date_col="date",
            ticker_col="ticker",
            seq_len=seq_len,
        )
        
        # 应该生成 30 - 10 + 1 = 21 个窗口
        expected_windows = 30 - seq_len + 1
        assert len(bundle.X_seq) == expected_windows
    
    def test_multiple_tickers_total_windows(self, sample_panel):
        """测试多个ticker总窗口数"""
        y = pd.Series(np.random.randn(len(sample_panel)), index=sample_panel.index)
        
        seq_len = 20
        bundle = build_panel_sequences(
            sample_panel[["date", "ticker", "feature1", "feature2"]],
            y,
            date_col="date",
            ticker_col="ticker",
            seq_len=seq_len,
        )
        
        # 每个ticker: 50 - 20 + 1 = 31 个窗口
        # 2个ticker: 31 * 2 = 62
        expected_windows = (50 - seq_len + 1) * 2
        assert len(bundle.X_seq) == expected_windows
    
    def test_meta_df_ticker_correct(self, sample_panel):
        """测试meta_df的ticker正确"""
        y = pd.Series(np.random.randn(len(sample_panel)), index=sample_panel.index)
        
        seq_len = 20
        bundle = build_panel_sequences(
            sample_panel[["date", "ticker", "feature1", "feature2"]],
            y,
            date_col="date",
            ticker_col="ticker",
            seq_len=seq_len,
        )
        
        # 检查ticker唯一值
        assert set(bundle.meta_df["ticker"].unique()) == {"A", "B"}
        
        # 检查每个ticker的窗口数
        for ticker in ["A", "B"]:
            ticker_count = len(bundle.meta_df[bundle.meta_df["ticker"] == ticker])
            assert ticker_count == 50 - seq_len + 1
    
    def test_meta_df_date_corresponds_to_last_day(self, sample_panel):
        """测试meta_df的日期对应窗口最后一天"""
        y = pd.Series(np.random.randn(len(sample_panel)), index=sample_panel.index)
        
        seq_len = 20
        bundle = build_panel_sequences(
            sample_panel[["date", "ticker", "feature1", "feature2"]],
            y,
            date_col="date",
            ticker_col="ticker",
            seq_len=seq_len,
        )
        
        # 检查第一个窗口的日期
        for ticker in ["A", "B"]:
            ticker_meta = bundle.meta_df[bundle.meta_df["ticker"] == ticker].reset_index(drop=True)
            ticker_data = sample_panel[sample_panel["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            
            # 第一个窗口的最后一天应该是seq_len-1
            assert ticker_meta.iloc[0]["date"] == ticker_data.iloc[seq_len - 1]["date"]
            
            # 第二个窗口的最后一天应该是seq_len
            assert ticker_meta.iloc[1]["date"] == ticker_data.iloc[seq_len]["date"]
