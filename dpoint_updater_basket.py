# dpoint_updater_basket.py
"""
Basket 模式每日更新器
====================

本模块提供 basket 模式的每日信号更新功能。

主要功能:
    - 加载 basket 数据
    - 重训或增量更新模型
    - 生成次日预测信号
    - 输出交易清单

使用示例:
    python dpoint_updater_basket.py --basket basket_1 --output ./signals
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from basket_loader import load_basket_folder
from constants import (
    DEFAULT_BASKET_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_TOP_K,
)
from excel_reporter import save_to_excel
from feature_dpoint import build_features_and_labels_panel
from panel_trainer import predict_panel, train_panel_model
from portfolio_builder import PortfolioConfig, build_portfolio
from utils import resolve_basket_path, set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Basket 模式每日 Dpoint 信号更新器"
    )

    # 数据参数
    parser.add_argument(
        "--basket",
        type=str,
        default=DEFAULT_BASKET_NAME,
        help=f"Basket 名称（默认：{DEFAULT_BASKET_NAME}）",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help=f"数据根目录（默认：{DEFAULT_DATA_ROOT}）",
    )
    parser.add_argument(
        "--basket_path",
        type=str,
        default=None,
        help="Basket 完整路径（优先级高于 --basket + --data_root）",
    )

    # 输出参数
    parser.add_argument(
        "--output",
        type=str,
        default="./signals",
        help="信号输出目录（默认：./signals）",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["csv", "parquet", "excel"],
        default="excel",
        help="输出格式（默认：excel）",
    )

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="已有模型路径（若指定则加载，否则重新训练）",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="强制重新训练模型",
    )

    # 组合参数
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"TopK 数量（默认：{DEFAULT_TOP_K}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）",
    )

    # 其他
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅验证数据，不生成信号",
    )

    return parser.parse_args()


def load_or_train_model(
    panel_df: pd.DataFrame,
    model_path: Optional[str],
    retrain: bool,
    seed: int,
) -> Tuple[Any, Dict[str, Any]]:
    """加载或训练模型"""
    if model_path and os.path.exists(model_path) and not retrain:
        logger.info(f"Loading model from: {model_path}")
        import joblib
        model = joblib.load(model_path)

        # 加载模型信息
        info_path = model_path.replace(".joblib", ".json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                model_info = json.load(f)
        else:
            model_info = {"loaded_from": model_path}

        return model, model_info
    else:
        logger.info("Training new model...")

        # 构建特征
        config = {
            "basket_name": "basket",
            "windows": [5, 10, 20, 60],
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_candle": True,
            "use_ta_indicators": True,
            "ta_windows": [6, 14, 20],
        }

        X, y, _ = build_features_and_labels_panel(
            panel_df,
            config,
            date_col="date",
            ticker_col="ticker",
            include_cross_section=True,
        )

        # 训练
        model_config = {
            "model_type": "xgboost",
            "model_params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
        }

        model, model_info = train_panel_model(
            X,
            y,
            model_config,
            date_col="date",
            ticker_col="ticker",
            seed=seed,
        )

        logger.info(f"Model trained with {model_info['n_features']} features")

        return model, model_info


def generate_signals(
    model: Any,
    panel_df: pd.DataFrame,
    top_k: int,
    trade_date: str,
) -> pd.DataFrame:
    """生成交易信号"""
    # 获取最新交易日
    latest_date = panel_df["date"].max()
    logger.info(f"Latest date in data: {latest_date}")

    # 构建最新特征
    config = {
        "basket_name": "basket",
        "windows": [5, 10, 20, 60],
        "use_momentum": True,
        "use_volatility": True,
        "use_volume": True,
        "use_candle": True,
        "use_ta_indicators": True,
    }

    # 只用最新一天的数据
    latest_df = panel_df[panel_df["date"] == latest_date].copy()

    if latest_df.empty:
        raise ValueError(f"No data found for latest date: {latest_date}")

    X, _, _ = build_features_and_labels_panel(
        latest_df,
        config,
        date_col="date",
        ticker_col="ticker",
        include_cross_section=False,  # 横截面需要多只股票
    )

    # 预测
    scores = predict_panel(
        model,
        X,
        date_col="date",
        ticker_col="ticker",
    )

    # 选择 TopK
    portfolio_config = PortfolioConfig(top_k=top_k, weighting="equal")
    portfolio = build_portfolio(
        scores,
        date=latest_date,
        config=portfolio_config,
        score_col="score",
        ticker_col="ticker",
        date_col="date",
    )

    # 生成信号
    signal_rows = []
    for i, ticker in enumerate(portfolio.tickers):
        signal_rows.append({
            "trade_date": trade_date,
            "ticker": ticker,
            "score": portfolio.scores[i],
            "weight": portfolio.weights[i],
            "action": "buy",
            "priority": i + 1,
        })

    signals_df = pd.DataFrame(signal_rows)
    return signals_df


def main() -> None:
    """主函数"""
    args = parse_args()

    # 设置种子
    set_global_seed(args.seed)

    # 解析 basket 路径
    if args.basket_path:
        basket_path = args.basket_path
    else:
        basket_path = resolve_basket_path(args.data_root, args.basket)

    logger.info(f"Basket path: {basket_path}")
    logger.info(f"Output dir: {args.output}")

    # 加载数据
    logger.info("Loading basket data...")
    panel_df, report, meta = load_basket_folder(basket_path)

    logger.info(f"Loaded {meta.n_tickers} tickers, {report.total_rows} rows")
    logger.info(f"Date range: {meta.date_range}")

    # Dry run
    if args.dry_run:
        logger.info("Dry run mode - data validation only")
        return

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载或训练模型
    model, model_info = load_or_train_model(
        panel_df,
        args.model_path,
        args.retrain,
        args.seed,
    )

    # 生成信号
    trade_date = datetime.now().strftime("%Y-%m-%d")
    signals_df = generate_signals(
        model,
        panel_df,
        args.top_k,
        trade_date,
    )

    logger.info(f"Generated {len(signals_df)} signals for {trade_date}")

    # 保存信号
    if args.output_format == "excel":
        output_path = os.path.join(args.output, f"signals_{trade_date}.xlsx")
        save_to_excel(
            output_path,
            scores_df=signals_df,
            config={"top_k": args.top_k, "basket": args.basket},
        )
    elif args.output_format == "csv":
        output_path = os.path.join(args.output, f"signals_{trade_date}.csv")
        signals_df.to_csv(output_path, index=False)
    elif args.output_format == "parquet":
        output_path = os.path.join(args.output, f"signals_{trade_date}.parquet")
        signals_df.to_parquet(output_path, index=False)

    logger.info(f"Signals saved to: {output_path}")

    # 保存模型
    import joblib
    model_dir = os.path.join(args.output, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{trade_date}.joblib")
    joblib.dump(model, model_path)

    info_path = model_path.replace(".joblib", ".json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    logger.info(f"Model saved to: {model_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
