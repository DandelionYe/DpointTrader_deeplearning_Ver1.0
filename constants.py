# constants.py
"""
全局常量集中管理。
所有模块从此处 import，避免跨模块依赖业务逻辑文件。
"""

# ==============================================================
# P1: Basket 多股票支持常量
# ==============================================================

# 默认数据根目录
DEFAULT_DATA_ROOT: str = "./data"

# 默认 Basket 名称
DEFAULT_BASKET_NAME: str = "basket_1"

# 默认文件匹配模式
DEFAULT_FILE_PATTERN: str = "*.csv"

# 数据契约版本
DATA_CONTRACT_VERSION: str = "1.0.0"

# 默认 CSV 列映射（从用户 CSV 列名映射到内部标准列名）
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "Date": "date",
    "Open (CNY, qfq)": "open_qfq",
    "High (CNY, qfq)": "high_qfq",
    "Low (CNY, qfq)": "low_qfq",
    "Close (CNY, qfq)": "close_qfq",
    "Volume (shares)": "volume",
    # 可选列
    "Amount (CNY)": "amount",
    "Turnover Rate": "turnover_rate",
}

# 单个标的 CSV 的必需核心列
REQUIRED_COLS_SINGLE: list[str] = [
    "date",
    "open_qfq",
    "high_qfq",
    "low_qfq",
    "close_qfq",
    "volume",
]

# 单个标的 CSV 的可选列
OPTIONAL_COLS: list[str] = [
    "amount",
    "turnover_rate",
]

# Panel 模式下的必需列
REQUIRED_COLS_PANEL: list[str] = [
    "date",
    "ticker",
    "open_qfq",
    "high_qfq",
    "low_qfq",
    "close_qfq",
    "volume",
]

# ==============================================================
# P1: 组合构建常量
# ==============================================================

# 默认 TopK 数量
DEFAULT_TOP_K: int = 5

# 默认调仓频率
DEFAULT_REBALANCE_FREQ: str = "daily"

# 默认单票权重上限
DEFAULT_MAX_WEIGHT: float = 0.20

# 默认现金缓冲比例
DEFAULT_CASH_BUFFER: float = 0.05

# 默认基准模式
DEFAULT_BENCHMARK_MODE: str = "equal_weight"

# 默认权重方式
DEFAULT_WEIGHTING: str = "equal"

# ==============================================================
# P1: 特征工程常量
# ==============================================================

# 默认标签模式
DEFAULT_LABEL_MODE: str = "binary_next_close_up"

# 默认是否包含横截面特征
DEFAULT_INCLUDE_CROSS_SECTION: bool = True

# ==============================================================
# P1: 回测常量
# ==============================================================

# 默认执行模式
DEFAULT_EXECUTION_MODE: str = "close_price"

# 默认涨跌停幅度
DEFAULT_LIMIT_UP_PCT: float = 0.10
DEFAULT_LIMIT_DOWN_PCT: float = 0.10

# 默认最小上市天数
DEFAULT_MIN_LISTING_DAYS: int = 60

# 默认过滤 ST 股
DEFAULT_FILTER_ST: bool = True
