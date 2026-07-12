"""The AI Mind · 第 1 章配套代码：学习问题（The Learning Problem）

把第 1 章正文里的 NumPy 实验沉淀为可导入、可测试的模块。

核心对象与正文一一对应：
    X, Y        输入/输出空间（此处 X=[0,1], Y=R）
    D           未知数据分布：y = f(x) + 高斯噪声
    H_d         假设空间：全体 d 次多项式（用最小二乘显式解 ERM）
    ell         损失：平方损失
    R_hat(h)    经验风险（训练集上的平均损失）——我们唯一算得出的量
    R(h)        真实风险（对 D 的期望，用密集网格数值积分近似）——我们真正在意的量

运行 `python -m ch01.learning_problem` 会打印正文那张 (degree, R_hat, R_true) 表。

数值可复现性说明（诚实版）：
    低次数（d<=5）在良态区，跨环境数值稳定。
    高次数（d>=6）用原始单项式 Vandermonde 基，系统接近病态，
    过拟合导致的 true risk 绝对量级（如 d=9 的 ~8287）对 BLAS/LAPACK 实现与
    NumPy 版本敏感——**趋势（先降后爆）稳定，绝对值不跨环境保证**。
    因此配套测试 test_ch01.py 对低次数用容差断言、对高次数只断言趋势。
"""
from __future__ import annotations

import numpy as np

# 平方损失 ℓ(ŷ, y) = (ŷ - y)^2 —— 写成可替换的函数对象，换任务只需替换它
def sq_loss(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (yhat - y) ** 2


# 未知目标 f：世界的隐藏规律，学习者看不到它，只看到带噪观测
def f(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x)


# 假设空间 H_d 的设计矩阵：d 次多项式的 Vandermonde 矩阵
def design(x: np.ndarray, d: int) -> np.ndarray:
    return np.vander(x, N=d + 1, increasing=True)


def make_dataset(seed: int = 0, n: int = 15, sigma: float = 0.15):
    """从分布 D 抽一个训练集 S = {(x_i, y_i)}。sigma 是噪声标准差，Bayes 风险下界 = sigma^2。

    刻意使用 NumPy 旧版全局 RNG（seed + uniform + randn），以与第 1 章正文的数值逐位一致。
    """
    np.random.seed(seed)
    x_train = np.sort(np.random.uniform(0.0, 1.0, size=n))
    y_train = f(x_train) + sigma * np.random.randn(n)
    return x_train, y_train


def fit_erm(x_train: np.ndarray, y_train: np.ndarray, d: int) -> np.ndarray:
    """经验风险最小化：解 argmin_w ||Phi w - y||^2（最小二乘），返回权重 w。"""
    Phi = design(x_train, d)
    w, *_ = np.linalg.lstsq(Phi, y_train, rcond=None)
    return w


def empirical_risk(w: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, d: int) -> float:
    """经验风险 R_hat(h) = (1/n) Σ ℓ(h(x_i), y_i)。"""
    return float(np.mean(sq_loss(design(x_train, d) @ w, y_train)))


def true_risk(w: np.ndarray, d: int, sigma: float = 0.15, n_grid: int = 2000) -> float:
    """真实风险 R(h) 的数值积分近似 + 不可消除的噪声地板 sigma^2（即 Bayes 风险下界）。"""
    x_grid = np.linspace(0.0, 1.0, n_grid)
    f_grid = f(x_grid)
    return float(np.mean(sq_loss(design(x_grid, d) @ w, f_grid)) + sigma**2)


def run_table(max_degree: int = 9, seed: int = 0, n: int = 15, sigma: float = 0.15):
    """复现正文的 (degree, R_hat, R_true) 表，返回 list[(d, R_hat, R_true)]。"""
    x_train, y_train = make_dataset(seed=seed, n=n, sigma=sigma)
    rows = []
    for d in range(0, max_degree + 1):
        w = fit_erm(x_train, y_train, d)
        rows.append((d, empirical_risk(w, x_train, y_train, d), true_risk(w, d, sigma=sigma)))
    return rows


def main() -> None:
    rows = run_table()
    print(f"{'degree':>6} | {'train MSE (R_hat)':>17} | {'true risk (R)':>14}")
    print("-" * 46)
    for d, rhat, rtrue in rows:
        print(f"{d:>6d} | {rhat:>17.4f} | {rtrue:>14.4f}")


if __name__ == "__main__":
    main()
