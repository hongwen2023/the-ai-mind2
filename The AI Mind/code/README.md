# The AI Mind · 配套代码

本目录承载课程正文里的可运行代码，并为每一处"招牌数值"提供回归测试，
兑现架构文档第 7 节对"可维护性/可复现"的承诺。

## 目录结构

```
code/
├── requirements.txt              # 锁定依赖（numpy==2.4.4）
├── ch01/
│   ├── __init__.py
│   └── learning_problem.py       # 第 1 章：学习问题（ERM / 经验风险 vs 真实风险 / 过拟合）
├── ch02/
│   ├── __init__.py
│   └── faces_of_intelligence.py  # 第 2 章：搜索一注（minimax 井字棋 + alpha-beta + 组合爆炸）
├── ch03/
│   ├── __init__.py
│   └── complexity_wall.py        # 第 3 章：多项式 vs 指数（斐波那契调用数 / 子集和枚举）
├── ch04/
│   ├── __init__.py
│   └── thinking_in_code.py       # 第 4 章：向量化（标准化/成对距离 loop vs vectorized）
└── tests/
    ├── test_ch01.py              # 第 1 章数值回归测试（低次容差断言 + 高次趋势断言）
    ├── test_ch02.py              # 第 2 章数值回归测试（确定性整数搜索，精确断言）
    ├── test_ch03.py              # 第 3 章数值回归测试（确定性计数，精确断言）
    └── test_ch04.py              # 第 4 章数值回归测试（loop vs 向量化等价，精确断言）
```

## 运行

```bash
# 1. 安装依赖（建议在虚拟环境中）
pip install -r requirements.txt

# 2. 复现第 1 章正文那张 (degree, R_hat, R_true) 表
python -m ch01.learning_problem        # 在 code/ 目录下运行

# 3. 跑测试
pytest                                 # 若已安装 pytest
python tests/test_ch01.py              # 无 pytest 时的兜底自检（打印 PASS/FAIL）
```

## 关于数值可复现性（诚实说明）

第 1 章那张表里，**低次数（d ≤ 5）在良态区，跨环境数值稳定**；
**高次数（d ≥ 6）**用原始单项式 Vandermonde 基，系统接近病态，
过拟合导致的 true risk 绝对量级（如 d=9 的 ~8287）**对 BLAS/LAPACK 实现与 NumPy 版本敏感**——
趋势（先降后爆）稳定，但绝对值不跨环境保证。

因此 `test_ch01.py` 的断言分两档：
- 低次数：对 `R_hat` / `R_true` 的绝对值做**容差断言**；
- 高次数：**只断言趋势**（R̂ 单调不增、R 最小点在 d=3、d≥6 灾难性放大、d=9 训练误差跌破噪声地板 σ²）。

这样既守住"过拟合戏剧成立"这一教学结论，又不对跨环境不稳定的数字做虚假承诺。
（这正是自我评估中指出的"8287 完全可复现是过度承诺"一条的工程回应。）
