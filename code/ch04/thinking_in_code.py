"""The AI Mind · 第 4 章配套代码：编程作为思维工具

把"同一个想法，两种表达"落成可运行、可断言的对照：
标准化 (standardization) 与成对欧氏距离 (pairwise distances) 各写两遍——
纯 Python 循环版（一次一个元素）与向量化 NumPy 版（一次一整块）——并断言两者结果一致。
证据以**操作/循环计数**为准（确定性、可断言）；浮点比较一律用 np.allclose 而非 ==。

标准差分母统一用 n（ddof=0，与 sklearn StandardScaler 一致），使两版严格对齐。

运行 `python -m ch04.thinking_in_code` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np


def make_data(seed: int = 42, n: int = 6, d: int = 3):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=5.0, scale=2.0, size=(n, d))


def standardize_loop(X):
    """纯 Python 循环版标准化。返回 (Z, 标准化步的标量赋值次数)。"""
    n, d = X.shape
    means = [0.0] * d
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += X[i, j]
        means[j] = s / n
    stds = [0.0] * d
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += (X[i, j] - means[j]) ** 2
        stds[j] = (s / n) ** 0.5
    Z = [[0.0] * d for _ in range(n)]
    n_scalar = 0
    for i in range(n):
        for j in range(d):
            Z[i][j] = (X[i, j] - means[j]) / stds[j]
            n_scalar += 1
    return np.array(Z), n_scalar


def standardize_vec(X):
    """向量化标准化。返回 Z。ddof=0（分母 n）以与循环版对齐。"""
    mu = X.mean(axis=0)      # (d,)
    sigma = X.std(axis=0)    # (d,) 默认 ddof=0
    return (X - mu) / sigma  # 广播 (n,d)-(d,) -> (n,d)


def pairwise_dist_loop(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(((X[i] - X[j]) ** 2).sum())
    return D


def pairwise_dist_vec(X):
    diff = X[:, None, :] - X[None, :, :]     # (n,1,d)-(1,n,d) -> (n,n,d)
    return np.sqrt((diff ** 2).sum(axis=2))  # (n,n)


def main() -> None:
    X = make_data()
    Z_loop, n_scalar = standardize_loop(X)
    Z_vec = standardize_vec(X)
    print("allclose:", np.allclose(Z_loop, Z_vec))
    print("循环版 Python 层标量赋值 (仅标准化那步):", n_scalar)
    print("向量化版 Python 层数组级操作 (mean/std/减/除): 4")
    print("标准化后列均值≈0:", np.allclose(Z_vec.mean(axis=0), 0.0))
    print("标准化后列标准差:", np.round(Z_vec.std(axis=0), 6))

    a = np.arange(3).reshape(3, 1)
    b = np.arange(4).reshape(1, 4)
    print("(3,1)+(1,4) 广播成:", (a + b).shape)
    print("成对距离 allclose:", np.allclose(pairwise_dist_vec(X), pairwise_dist_loop(X)))


if __name__ == "__main__":
    main()
