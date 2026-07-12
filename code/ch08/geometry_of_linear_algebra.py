"""The AI Mind · 第 8 章配套代码：线性代数——表示与变换的几何

两个演示把"矩阵是变换"变成可核验的事实：
  A. 矩阵把单位圆变成椭圆——奇异值 (singular value) 就是椭圆半轴长（最大/最小拉伸）。
     对称矩阵时 U=V，椭圆主轴落在特征向量方向上。
  B. 低秩逼近 (low-rank approximation) 与 Eckart–Young 定理：
     秩-k 逼近是最优的，其 Frobenius 误差 = sqrt(丢弃奇异值的平方和)。

纯 NumPy、固定种子，输出确定。
运行 `python -m ch08.geometry_of_linear_algebra` 打印正文那两组可核验输出。
"""
from __future__ import annotations

import numpy as np


def ellipse_demo(A):
    """矩阵 A 作用单位圆。返回 (奇异值 S, 像的最大长度, 像的最小长度, U, Vt)。"""
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle
    U, S, Vt = np.linalg.svd(A)
    norms = np.linalg.norm(ellipse, axis=0)
    return S, float(norms.max()), float(norms.min()), U, Vt


def make_low_rank_matrix(seed=0, m=20, n=15, r=3, noise=0.01):
    """造一个'本质 r 维'的数据矩阵：(m×r)(r×n) + 噪声。"""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    return M + noise * rng.standard_normal((m, n))


def low_rank(U, S, Vt, k):
    """只保留最大的 k 个奇异值的秩-k 逼近。"""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def eckart_young_table(M, kmax=6):
    """返回 [(k, frob误差, sqrt(尾部σ平方和))]，验证 Eckart–Young。"""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    rows = []
    for k in range(1, kmax + 1):
        err = float(np.linalg.norm(M - low_rank(U, S, Vt, k), "fro"))
        ey = float(np.sqrt(np.sum(S[k:] ** 2)))
        rows.append((k, err, ey))
    return S, rows


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    S, mx, mn, U, Vt = ellipse_demo(A)
    print("演示一：单位圆 -> 椭圆")
    print("奇异值:", S, " 像最大长度:", round(mx, 4), " 最小长度:", round(mn, 4))
    print("对称阵 U 与 V 相同:", np.allclose(U, Vt.T))

    M = make_low_rank_matrix()
    S, rows = eckart_young_table(M)
    print("\n演示二：低秩逼近 + Eckart–Young")
    print("奇异值:", S)
    print(" k |  frob误差  | sqrt(尾部σ平方和)")
    for k, err, ey in rows:
        print(f" {k} | {err:9.6f} | {ey:9.6f}")


if __name__ == "__main__":
    main()
