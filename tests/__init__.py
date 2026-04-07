# tests/__init__.py
"""
Basket 模式测试套件
==================

本测试套件包含以下测试模块：

1. test_basket_loader.py - Basket 加载器测试
2. test_panel_builder.py - Panel 构建器测试
3. test_portfolio_builder.py - 组合构建器测试
4. test_ranking_metrics.py - 排序指标测试
5. test_cross_sectional_features.py - 横截面特征测试
6. test_basket_smoke.py - Basket 冒烟测试
7. test_end_to_end.py - 端到端集成测试

运行所有测试:
    pytest tests/ -v

运行特定测试:
    pytest tests/test_basket_loader.py -v
    pytest tests/test_end_to_end.py -v
"""
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
