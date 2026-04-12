from typing import Optional, Set

import pandas as pd

from feature_dpoint import build_features_and_labels_panel


def _make_price_panel(
    dates: pd.DatetimeIndex,
    *,
    ticker: str,
    missing_dates: Optional[Set[pd.Timestamp]] = None,
) -> pd.DataFrame:
    missing_dates = missing_dates or set()
    rows = []
    for idx, date in enumerate(dates):
        if date in missing_dates:
            continue
        close = 10.0 + idx
        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "open_qfq": close * 0.99,
                "high_qfq": close * 1.01,
                "low_qfq": close * 0.98,
                "close_qfq": close,
                "volume": 1_000_000 + idx * 1000,
            }
        )
    return pd.DataFrame(rows)


def _feature_config() -> dict:
    return {
        "basket_name": "test",
        "windows": [2],
        "use_momentum": False,
        "use_volatility": False,
        "use_volume": False,
        "use_candle": False,
        "use_ta_indicators": False,
    }


def test_max_label_date_filters_training_tail_for_next_day_label():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    panel_df = _make_price_panel(dates, ticker="A")
    retrain_date = dates[2]

    X, y, _, label_meta = build_features_and_labels_panel(
        panel_df,
        _feature_config(),
        date_col="date",
        ticker_col="ticker",
        include_cross_section=False,
        label_horizon_days=1,
        max_label_date=retrain_date,
        return_label_end_date=True,
    )

    assert not X.empty
    assert len(X) == len(y) == len(label_meta)
    assert (label_meta["label_end_date"] <= retrain_date).all()
    assert retrain_date not in set(X["date"])


def test_longer_label_horizon_removes_last_horizon_samples():
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    panel_df = _make_price_panel(dates, ticker="A")
    retrain_date = dates[7]

    X, _, _, label_meta = build_features_and_labels_panel(
        panel_df,
        _feature_config(),
        date_col="date",
        ticker_col="ticker",
        include_cross_section=False,
        label_horizon_days=5,
        max_label_date=retrain_date,
        return_label_end_date=True,
    )

    assert not X.empty
    assert X["date"].max() == dates[2]
    assert label_meta["label_end_date"].max() == retrain_date


def test_label_end_date_advances_by_each_ticker_trading_calendar():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    panel_df = pd.concat(
        [
            _make_price_panel(dates, ticker="A"),
            _make_price_panel(dates, ticker="B", missing_dates={dates[3]}),
        ],
        ignore_index=True,
    )

    _, _, _, label_meta = build_features_and_labels_panel(
        panel_df,
        _feature_config(),
        date_col="date",
        ticker_col="ticker",
        include_cross_section=False,
        label_horizon_days=1,
        return_label_end_date=True,
    )

    ticker_b = label_meta[label_meta["ticker"] == "B"].set_index("date").sort_index()
    assert ticker_b.loc[dates[2], "label_end_date"] == dates[4]
