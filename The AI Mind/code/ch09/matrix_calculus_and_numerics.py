"""The AI Mind · 第 9 章配套代码：矩阵微分与数值基础

三个演示：
  A. 梯度检验 (gradient check)：把本章手推的解析梯度 (1/n)Σ(σ(w·x)-y)x
     与有限差分数值梯度对照 —— 既证明又数值验证 Ch7 那张欠条。
  B. softmax 数值稳定性：天真 softmax 对大 logit 溢出成 nan；
     log-sum-exp（先减最大值）数学等价但数值安全。
  C. 条件数 (condition number)：范德蒙德 (Vandermonde) 矩阵随阶数病态爆炸
     （回收 Ch5 的伏笔：κ 到 1e15 量级）。

纯 NumPy、固定种子。
运行 `python -m ch09.matrix_calculus_and_numerics` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def loss(w, X, y):
    p = sigmoid(X @ w)
    eps = 1e-12  # 回指 Ch7：防 log(0)
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def grad_analytic(w, X, y):
    """本章推出的解析梯度：(1/n)Σ(σ(w·x)-y)x。"""
    p = sigmoid(X @ w)
    return X.T @ (p - y) / len(y)


def grad_numeric(w, X, y, eps=1e-6):
    """中心差分数值梯度。"""
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w); e[j] = eps
        g[j] = (loss(w + e, X, y) - loss(w - e, X, y)) / (2 * eps)
    return g


def gradient_check(seed=42, n=50, d=5):
    """返回解析梯度与数值梯度的相对误差。"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    y = (sigmoid(X @ w_true) > rng.uniform(size=n)).astype(float)
    w = rng.standard_normal(d)
    g_ana = grad_analytic(w, X, y)
    g_num = grad_numeric(w, X, y)
    return float(np.linalg.norm(g_ana - g_num) / np.linalg.norm(g_ana + g_num))


def softmax_naive(z):
    e = np.exp(z)
    return e / np.sum(e)


def softmax_stable(z):
    z = z - np.max(z)   # log-sum-exp 核心：先减最大值
    e = np.exp(z)
    return e / np.sum(e)


def vandermonde_cond(m):
    xs = np.linspace(0, 1, m)
    V = np.vander(xs, increasing=True)
    return float(np.linalg.cond(V))


def main() -> None:
    print("梯度检验相对误差:", gradient_check())
    big = np.array([1000., 1001., 1002.])
    with np.errstate(over="ignore", invalid="ignore"):
        print("天真版(大logit):", softmax_naive(big))
    print("稳定版(大logit):", softmax_stable(big), " 和 =", softmax_stable(big).sum())
    small = np.array([1., 2., 3.])
    print("小输入两者一致:", np.allclose(softmax_naive(small), softmax_stable(small)))
    for m in (5, 10, 15, 19):
        print(f"Vandermonde m={m:2d} 条件数 = {vandermonde_cond(m):.3e}")


if __name__ == "__main__":
    main()
