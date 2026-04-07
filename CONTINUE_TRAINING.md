# 🔄 续训功能使用指南

## 📋 什么是续训模式？

续训模式允许你在已有实验结果的基础上继续训练，无需从头开始。适用于：
- 想增加搜索次数优化模型
- 想在原实验基础上测试新参数
- 想快速重新回测已有模型

## 🚀 使用方法

### 方式 1：从最新实验续训

```bash
# 从最新的实验目录续训
python main_basket.py --basket basket_1 --continue_from latest --additional_runs 50
```

### 方式 2：从指定实验续训

```bash
# 从指定的实验目录续训
python main_basket.py --basket basket_1 --continue_from ./output_basket/exp_001 --additional_runs 100
```

### 方式 3：仅回测已有模型（不训练）

```bash
# 加载已有模型，直接回测
python main_basket.py --basket basket_1 --continue_from ./output_basket/exp_001 --runs 0
```

## 📊 续训参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--continue_from` | 续训来源 | `None` | `latest` 或 `./output_basket/exp_001` |
| `--additional_runs` | 额外搜索次数 | `50` | `50`, `100`, `200` |
| `--runs` | 总搜索次数 | `50` | 续训时会被忽略 |

## 📁 续训流程

```
1. 加载已有实验的 manifest.json
   └─→ 读取最优配置 (best_config)
   
2. 加载已有模型文件 (.joblib)
   └─→ 跳过训练阶段
   
3. 使用已有模型进行预测和回测
   └─→ 生成新的结果文件
   
4. 更新实验目录
   └─→ 保留历史记录
```

## 💡 使用示例

### 示例 1：增加搜索次数

```bash
# 第一次运行 50 次搜索
python main_basket.py --basket basket_1 --runs 50

# 对结果不满意，再增加 100 次搜索
python main_basket.py --basket basket_1 --continue_from latest --additional_runs 100
```

### 示例 2：更换数据后快速回测

```bash
# 更新数据后，使用已有模型快速回测
python main_basket.py --basket basket_1 --continue_from ./output_basket/exp_001
```

### 示例 3：对比多个实验

```bash
# 运行多个实验
python main_basket.py --basket basket_1 --runs 50  # exp_001
python main_basket.py --basket basket_1 --runs 50  # exp_002
python main_basket.py --basket basket_1 --runs 50  # exp_003

# 对比结果
python compare_runs.py --output_dir ./output_basket
```

## ⚠️ 注意事项

1. **数据一致性**：续训时应使用与之前相同的数据集
2. **配置兼容**：确保特征配置与之前一致
3. **目录权限**：续训会修改原实验目录，确保有写权限
4. **模型版本**：不同版本的模型文件可能不兼容

## 🔍 查看实验历史

```bash
# 列出所有实验
dir output_basket\exp_*

# 查看实验详情
type output_basket\exp_001\manifest.json
```

## 📊 输出文件

续训后，实验目录会包含：

```
output_basket/exp_001/
├── manifest.json         # 实验元数据（更新）
├── results.xlsx          # Excel 结果（更新）
├── report.html           # HTML 报告（更新）
├── models/
│   ├── model_2024-01-01.joblib  # 原模型
│   └── model_2024-01-02.joblib  # 新模型（如果有）
└── artifacts/            # 其他 artifacts
```

---

**最后更新**: 2026-03-22
**版本**: Continue Training v1.0
