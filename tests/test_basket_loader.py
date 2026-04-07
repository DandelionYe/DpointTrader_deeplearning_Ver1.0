import os
import shutil
import sys
import tempfile
import uuid

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basket_loader import (
    BasketMeta,
    BasketReport,
    discover_basket_files,
    extract_ticker_from_filename,
    load_basket_folder,
)
from csv_loader import SingleStockReport, load_single_csv, standardize_columns, validate_csv_structure


@pytest.fixture
def local_tmpdir():
    path = os.path.join(".local", "tmp", "basket_loader", str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class TestStandardizeColumns:
    def test_default_mapping(self):
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "Open (CNY, qfq)": [10.0, 10.5],
                "Close (CNY, qfq)": [10.2, 10.8],
            }
        )
        result = standardize_columns(df)
        assert "date" in result.columns
        assert "open_qfq" in result.columns
        assert "close_qfq" in result.columns

    def test_custom_mapping(self):
        df = pd.DataFrame({"DATE": ["2024-01-01"], "CLOSE": [10.0]})
        result = standardize_columns(df, column_map={"DATE": "date", "CLOSE": "close_qfq"})
        assert "date" in result.columns
        assert "close_qfq" in result.columns


class TestValidateCsvStructure:
    def test_valid_structure(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "open_qfq": [10.0],
                "close_qfq": [10.2],
                "high_qfq": [10.5],
                "low_qfq": [9.8],
                "volume": [1000],
            }
        )
        valid, missing_required, _ = validate_csv_structure(df)
        assert valid is True
        assert len(missing_required) == 0

    def test_missing_required_columns(self):
        df = pd.DataFrame({"date": ["2024-01-01"], "open_qfq": [10.0]})
        valid, missing_required, _ = validate_csv_structure(df, strict=False)
        assert valid is False
        assert "close_qfq" in missing_required


class TestExtractTickerFromFilename:
    def test_standard_filename(self):
        assert extract_ticker_from_filename("600036.csv") == "600036"
        assert extract_ticker_from_filename("002517_20101207.csv") == "002517_20101207"

    def test_path_with_directory(self):
        path = os.path.join("data", "basket_1", "600036.csv")
        assert extract_ticker_from_filename(path) == "600036"


class TestDiscoverBasketFiles:
    def test_discover_csv_files(self, local_tmpdir):
        for ticker in ["600036", "601318", "002517"]:
            pd.DataFrame({"date": ["2024-01-01"], "close": [10.0]}).to_csv(
                os.path.join(local_tmpdir, f"{ticker}.csv"),
                index=False,
            )
        files = discover_basket_files(local_tmpdir, file_pattern="*.csv")
        assert len(files) == 3
        assert all(path.endswith(".csv") for path in files)


class TestLoadSingleCsv:
    def test_load_valid_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write('"Date","Open (CNY, qfq)","High (CNY, qfq)","Low (CNY, qfq)","Close (CNY, qfq)","Volume (shares)"\n')
            f.write("2024-01-01,10.0,10.5,9.8,10.2,1000\n")
            f.write("2024-01-02,10.2,10.8,10.0,10.5,1200\n")
            temp_path = f.name

        try:
            df, report = load_single_csv(temp_path, ticker="TEST")
            assert len(df) == 2
            assert report.ticker == "TEST"
            assert report.rows_raw == 2
            assert "date" in df.columns
            assert "close_qfq" in df.columns
        finally:
            os.unlink(temp_path)

    def test_load_with_derived_amount(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write('"Date","Open (CNY, qfq)","High (CNY, qfq)","Low (CNY, qfq)","Close (CNY, qfq)","Volume (shares)"\n')
            f.write("2024-01-01,10.0,10.5,9.8,10.2,1000\n")
            temp_path = f.name

        try:
            df, report = load_single_csv(temp_path)
            assert "amount" in df.columns
            assert len(report.derived_cols) > 0
        finally:
            os.unlink(temp_path)


class TestLoadBasketFolder:
    def test_load_basket_folder(self, local_tmpdir):
        for i, ticker in enumerate(["600036", "601318"]):
            df = pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=5),
                    "Open (CNY, qfq)": [10.0 + i * 0.1] * 5,
                    "High (CNY, qfq)": [10.5 + i * 0.1] * 5,
                    "Low (CNY, qfq)": [9.8 + i * 0.1] * 5,
                    "Close (CNY, qfq)": [10.2 + i * 0.1] * 5,
                    "Volume (shares)": [1000 + i * 100] * 5,
                }
            )
            df.to_csv(os.path.join(local_tmpdir, f"{ticker}.csv"), index=False)

        panel_df, report, meta = load_basket_folder(local_tmpdir)

        assert isinstance(meta, BasketMeta)
        assert isinstance(report, BasketReport)
        assert meta.n_tickers == 2
        assert report.total_rows == 10
        assert "ticker" in panel_df.columns
        assert "date" in panel_df.columns
        assert panel_df["ticker"].nunique() == 2

    def test_load_basket_folder_outer_panel_keeps_sparse_dates(self, local_tmpdir):
        df_a = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "Open (CNY, qfq)": [10.0, 10.1],
                "High (CNY, qfq)": [10.2, 10.3],
                "Low (CNY, qfq)": [9.8, 9.9],
                "Close (CNY, qfq)": [10.1, 10.2],
                "Volume (shares)": [1000, 1000],
            }
        )
        df_b = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "Open (CNY, qfq)": [20.0, 20.1],
                "High (CNY, qfq)": [20.2, 20.3],
                "Low (CNY, qfq)": [19.8, 19.9],
                "Close (CNY, qfq)": [20.1, 20.2],
                "Volume (shares)": [2000, 2000],
            }
        )
        df_a.to_csv(os.path.join(local_tmpdir, "A.csv"), index=False)
        df_b.to_csv(os.path.join(local_tmpdir, "B.csv"), index=False)

        panel_df, _, _ = load_basket_folder(local_tmpdir)

        assert panel_df["date"].nunique() == 3
        assert len(panel_df) == 4
