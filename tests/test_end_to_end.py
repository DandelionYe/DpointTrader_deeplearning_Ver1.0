# test_end_to_end.py
"""
端到端集成测试
==============

测试完整的 basket 流程：
1. 加载 basket 数据
2. 构建特征和标签
3. 训练模型
4. 生成预测
5. 构建组合
6. 回测
7. 输出结果
"""
import os
import sys
import pandas as pd
import numpy as np


def test_end_to_end():
    """完整流程测试"""
    print("=" * 60)
    print("Basket 模式端到端测试")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/7] 加载 basket 数据...")
    from basket_loader import load_basket_folder
    
    basket_path = "data/basket_1"
    panel_df, report, meta = load_basket_folder(basket_path)
    
    print(f"  Basket: {meta.basket_name}")
    print(f"  Tickers: {meta.n_tickers}")
    print(f"  Total rows: {report.total_rows}")
    print(f"  Date range: {meta.date_range}")
    assert meta.n_tickers == 3, f"Expected 3 tickers, got {meta.n_tickers}"
    print("  ✓ 数据加载成功")
    
    # 2. 验证 panel
    print("\n[2/7] 验证 panel 结构...")
    from panel_builder import validate_panel
    
    valid, issues = validate_panel(panel_df)
    # 稀疏警告可以忽略（不同股票有不同的上市日期）
    critical_issues = [i for i in issues if "sparse" not in i.lower()]
    if critical_issues:
        raise AssertionError(f"Panel 验证失败：{critical_issues}")
    print(f"  Panel shape: {panel_df.shape}")
    print(f"  Columns: {panel_df.columns.tolist()}")
    if issues:
        print(f"  Notes: {issues}")
    print("  ✓ Panel 验证通过")
    
    # 3. 构建特征和标签
    print("\n[3/7] 构建特征和标签...")
    from feature_dpoint import build_features_and_labels_panel
    
    feature_config = {
        "basket_name": "test",
        "windows": [5, 10, 20],
        "use_momentum": True,
        "use_volatility": True,
        "use_volume": True,
        "use_candle": True,
        "use_ta_indicators": False,
    }
    
    X, y, feature_meta = build_features_and_labels_panel(
        panel_df,
        feature_config,
        date_col="date",
        ticker_col="ticker",
        include_cross_section=True,
    )
    
    print(f"  Features: {len(feature_meta.feature_names)}")
    print(f"  Samples: {feature_meta.n_samples}")
    print(f"  Tickers: {feature_meta.n_tickers}")
    assert len(feature_meta.feature_names) > 10, "特征数量过少"
    print("  ✓ 特征构建成功")
    
    # 4. 生成日期切分
    print("\n[4/7] 生成 walk-forward 切分...")
    from splitters import walkforward_splits_by_date
    
    splits = walkforward_splits_by_date(
        X,
        date_col="date",
        ticker_col="ticker",
        n_folds=2,
        train_start_ratio=0.5,
    )
    
    print(f"  Number of folds: {len(splits)}")
    assert len(splits) >= 1, "没有生成有效的切分"
    print("  ✓ 切分生成成功")
    
    # 5. 训练模型
    print("\n[5/7] 训练模型...")
    from panel_trainer import train_panel_model
    
    model_config = {
        "model_type": "xgb",  # 使用 models.py 中的模型类型名称
        "model_params": {
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.1,
        },
    }
    
    # 使用第一个切分训练
    train_dates, val_dates = splits[0]
    train_mask = X["date"].isin(train_dates)
    val_mask = X["date"].isin(val_dates)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples: {len(X_val)}")
    
    model, model_info = train_panel_model(
        X_train,
        y_train,
        model_config,
        date_col="date",
        ticker_col="ticker",
        seed=42,
    )
    
    print(f"  Model type: {model_info['model_type']}")
    print(f"  Features: {model_info['n_features']}")
    print("  ✓ 模型训练成功")
    
    # 6. 生成预测
    print("\n[6/7] 生成预测...")
    from panel_trainer import predict_panel
    from ranking_metrics import compute_rank_ic
    
    val_pred = predict_panel(
        model,
        X_val,
        date_col="date",
        ticker_col="ticker",
    )
    val_pred["label"] = y_val.values
    
    print(f"  Predictions: {len(val_pred)}")
    
    # 计算 RankIC
    rank_ic = compute_rank_ic(
        val_pred,
        score_col="score",
        label_col="label",
        date_col="date",
    )
    print(f"  RankIC mean: {rank_ic.mean():.4f}")
    print("  ✓ 预测生成成功")
    
    # 7. 构建组合并回测
    print("\n[7/7] 构建组合并回测...")
    from portfolio_builder import PortfolioConfig, build_portfolio
    from backtester_engine import backtest_from_scores
    
    portfolio_config = PortfolioConfig(
        top_k=2,
        weighting="equal",
        max_weight=0.5,
        cash_buffer=0.05,
    )
    
    # 简化回测：只用验证集数据
    backtest_result = backtest_from_scores(
        panel_df[panel_df["date"].isin(val_dates)],
        val_pred,
        portfolio_config=portfolio_config,
        score_col="score",
        ticker_col="ticker",
        date_col="date",
        initial_cash=100000.0,
    )
    
    print(f"  Equity curve rows: {len(backtest_result.equity_curve)}")
    print(f"  Total trades: {len(backtest_result.trades)}")
    
    if not backtest_result.equity_curve.empty:
        initial = backtest_result.equity_curve["equity"].iloc[0]
        final = backtest_result.equity_curve["equity"].iloc[-1]
        print(f"  Initial equity: {initial:,.2f}")
        print(f"  Final equity: {final:,.2f}")
        print(f"  Return: {(final-initial)/initial:.2%}")
    
    print("  ✓ 回测完成")
    
    # 8. 保存结果
    print("\n[8/8] 保存结果...")
    from excel_reporter import save_to_excel
    from html_reporter import generate_html_report
    
    output_dir = "output_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Excel
    excel_path = os.path.join(output_dir, "test_results.xlsx")
    save_to_excel(
        excel_path,
        equity_curve=backtest_result.equity_curve,
        trades=backtest_result.trades,
        scores_df=val_pred,
        config={"test": "end_to_end"},
        metrics={"rank_ic": float(rank_ic.mean())},
    )
    print(f"  Excel: {excel_path}")
    
    # HTML
    html_path = os.path.join(output_dir, "test_report.html")
    generate_html_report(
        html_path,
        equity_curve=backtest_result.equity_curve,
        metrics={"rank_ic_mean": float(rank_ic.mean())},
        config={"test": "end_to_end"},
    )
    print(f"  HTML: {html_path}")
    
    print("  ✓ 结果保存成功")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_end_to_end()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
