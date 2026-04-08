# html_reporter.py
"""
HTML 报告生成器
================

本模块提供生成 HTML 格式回测报告的功能。

主要功能:
    - generate_html_report: 生成 HTML 报告
    - 包含组合表现、回撤、IC、个股贡献等

使用示例:
    >>> from html_reporter import generate_html_report
    >>> generate_html_report("report.html", backtest_result, metrics)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def generate_html_report(
    output_path: str,
    *,
    equity_curve: Optional[pd.DataFrame] = None,
    benchmark_curve: Optional[pd.DataFrame] = None,
    execution_stats: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    basket_info: Optional[Dict[str, Any]] = None,
    notes: Optional[List[str]] = None,
) -> str:
    """
    生成 HTML 报告。

    Args:
        output_path: 输出文件路径
        equity_curve: 权益曲线 DataFrame
        metrics: 指标字典
        config: 配置字典
        basket_info: Basket 信息
        notes: 注释列表

    Returns:
        生成的 HTML 内容
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dpoint Trader - Basket 回测报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Dpoint Trader - Basket 回测报告</h1>
        <p>生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
"""

    # Basket 信息
    if basket_info:
        html += """
        <h2>📁 Basket 信息</h2>
        <table>
            <tr><th>属性</th><th>值</th></tr>
"""
        for key, value in basket_info.items():
            html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
        html += """        </table>
"""

    # 组合表现指标
    if metrics:
        html += """
        <h2>📈 组合表现</h2>
        <div class="metrics-grid">
"""
        metric_order = [
            ("total_return", "总收益率", "{:.2%}"),
            ("annual_return", "年化收益", "{:.2%}"),
            ("sharpe", "夏普比率", "{:.2f}"),
            ("max_drawdown", "最大回撤", "{:.2%}"),
            ("win_rate", "胜率", "{:.2%}"),
            ("avg_trade_return", "平均交易收益", "{:.2%}"),
        ]

        for key, label, fmt in metric_order:
            value = metrics.get(key)
            if value is not None:
                try:
                    num_value = float(value)
                    formatted = fmt.format(num_value)
                    css_class = "positive" if num_value > 0 else "negative" if num_value < 0 else ""
                    html += f"""            <div class="metric-card">
                <div class="metric-value {css_class}">{formatted}</div>
                <div class="metric-label">{label}</div>
            </div>
"""
                except (ValueError, TypeError):
                    pass

        # 其他指标
        other_metrics = [
            ("rank_ic_mean", "RankIC 均值"),
            ("rank_ic_std", "RankIC 标准差"),
            ("ic_mean", "IC 均值"),
            ("topk_return_annual", "TopK 年化收益"),
            ("n_trades", "交易次数"),
            ("n_tickers", "持仓股票数"),
        ]

        for key, label in other_metrics:
            value = metrics.get(key)
            if value is not None:
                html += f"""            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
"""

        html += """        </div>
"""

    # 权益曲线图表（如果有）
    if equity_curve is not None and not equity_curve.empty:
        html += """
        <h2>📉 权益曲线</h2>
        <p>权益曲线数据已导出到 Excel 文件的 "PortfolioEquity" sheet</p>
"""
        # 简单统计
        if "equity" in equity_curve.columns and "cum_return" in equity_curve.columns:
            initial = equity_curve["equity"].iloc[0]
            final = equity_curve["equity"].iloc[-1]
            total_return = (final - initial) / initial
            max_equity = equity_curve["equity"].max()
            min_equity = equity_curve["equity"].min()
            max_dd = ((equity_curve["equity"] - equity_curve["equity"].cummax()) / equity_curve["equity"].cummax()).min()

            html += f"""
        <table>
            <tr><th>统计项</th><th>值</th></tr>
            <tr><td>初始权益</td><td>{initial:,.2f}</td></tr>
            <tr><td>最终权益</td><td>{final:,.2f}</td></tr>
            <tr><td>总收益</td><td class="{'positive' if total_return > 0 else 'negative' if total_return < 0 else ''}">{total_return:.2%}</td></tr>
            <tr><td>最高权益</td><td>{max_equity:,.2f}</td></tr>
            <tr><td>最低权益</td><td>{min_equity:,.2f}</td></tr>
            <tr><td>最大回撤</td><td class="negative">{max_dd:.2%}</td></tr>
        </table>
"""

    # 配置信息
    if benchmark_curve is not None and not benchmark_curve.empty and "bnh_cum_return" in benchmark_curve.columns:
        benchmark_total_return = float(benchmark_curve["bnh_cum_return"].iloc[-1])
        html += f"""
        <h2>Benchmark</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Benchmark total return</td><td>{benchmark_total_return:.2%}</td></tr>
        </table>
"""

    if execution_stats:
        html += """
        <h2>Execution Stats</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""
        for key, value in execution_stats.items():
            if key == "reject_reasons":
                continue
            html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
        html += """        </table>
"""
        reject_reasons = execution_stats.get("reject_reasons", {}) or {}
        if reject_reasons:
            html += """
        <h2>Reject Reasons</h2>
        <table>
            <tr><th>Reason</th><th>Count</th></tr>
"""
            for reason, count in reject_reasons.items():
                html += f"            <tr><td>{reason}</td><td>{count}</td></tr>\n"
            html += """        </table>
"""

    if config:
        html += """
        <h2>⚙️ 配置信息</h2>
        <table>
            <tr><th>参数</th><th>值</th></tr>
"""
        for key, value in config.items():
            if key not in ["feature_config", "model_config"]:  # 跳过复杂配置
                html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
        html += """        </table>
"""

    # 注释
    if notes:
        html += """
        <h2>📝 注释</h2>
        <ul>
"""
        for note in notes:
            html += f"            <li>{note}</li>\n"
        html += """        </ul>
"""

    # 页脚
    html += f"""
        <div class="footer">
            <p>报告由 Dpoint Trader 生成 | 数据契约版本：1.0.0</p>
        </div>
    </div>
</body>
</html>
"""

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Generated HTML report: {output_path}")
    return html


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "generate_html_report",
]
