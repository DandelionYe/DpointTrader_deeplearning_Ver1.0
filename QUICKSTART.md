# 🚀 Dpoint Trader Basket 模式 - 快速启动指南

## ✅ 修复完成

项目已经修复，现在可以正常运行了！

## 📋 运行步骤

### 1. 激活 Conda 环境
```bash
conda activate ashare_dpoint
```

### 2. 运行 Basket 回测
```bash
cd J:\DpointTrader_deeplearning_Ver1.0

# 基本命令
python main_basket.py --basket basket_1 --data_root ./data --runs 50

# 完整参数示例
python main_basket.py ^
    --basket basket_1 ^
    --data_root ./data ^
    --runs 100 ^
    --seed 42 ^
    --top_k 5 ^
    --weighting equal ^
    --rebalance_freq daily ^
    --max_weight 0.20 ^
    --initial_cash 100000 ^
    --n_folds 4 ^
    --output_dir ./output_basket
```

### 3. 查看结果

运行完成后，结果保存在：
```
output_basket/exp_001/
├── results.xlsx          # Excel 结果
├── report.html           # HTML 报告
├── manifest.json         # 实验元数据
└── models/               # 模型文件
```

打开 HTML 报告：
```bash
start output_basket\exp_001\report.html
```

## 📊 测试结果

刚才的运行结果：
- **初始资金**: 100,000 元
- **最终权益**: 143,881.13 元
- **总收益**: +43.8%
- **交易次数**: 5,566 次
- **RankIC**: 0.0569

## 🔧 常用参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--basket` | Basket 文件夹名 | `basket_1` | - |
| `--data_root` | 数据根目录 | `./data` | - |
| `--runs` | 搜索迭代次数 | `50` | `50-200` |
| `--top_k` | 持仓股票数 | `5` | `3-10` |
| `--weighting` | 权重方式 | `equal` | `equal/score` |
| `--n_folds` | Walk-forward 折数 | `4` | `4-6` |
| `--initial_cash` | 初始资金 | `100000` | - |

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_end_to_end.py -v
```

## ⚠️ 常见问题

### Q1: Panel is XX% sparse 警告
这是**正常现象**，不同股票有不同的上市日期。程序会自动处理。

### Q2: 找不到数据文件
确保 `data/basket_1/` 目录存在且包含 CSV 文件：
```bash
dir data\basket_1\*.csv
```

### Q3: CSV 格式错误
CSV 列名必须使用引号包裹：
```csv
"Date","Open (CNY, qfq)","High (CNY, qfq)",...
```

### Q4: 内存不足
减少搜索次数和股票数量：
```bash
python main_basket.py --basket basket_1 --runs 20 --top_k 3
```

## 📁 数据准备

将股票 CSV 文件放入 `data/basket_1/` 目录：
```
data/basket_1/
├── 600036.csv
├── 601318.csv
└── 002517.csv
```

CSV 格式示例：
```csv
Date,Open (CNY, qfq),High (CNY, qfq),Low (CNY, qfq),Close (CNY, qfq),Volume (shares)
2024-01-01,10.0,10.5,9.8,10.2,1000000
2024-01-02,10.2,10.8,10.0,10.5,1200000
```

## 📞 获取帮助

```bash
# 查看帮助
python main_basket.py --help
```

---

**最后更新**: 2026-03-22
**版本**: Basket Mode v1.0
