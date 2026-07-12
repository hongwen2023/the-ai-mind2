"""The AI Mind · 第 17 章配套代码：核方法与 SVM

在非线性可分的同心圆数据上，把核技巧与 SVM 的核心事实跑成可核验的结果：
  - **核技巧**：线性核 SVM 近乎随机（画不出弯边界），RBF 核 SVM 完美分类——
    全程只算核 k(x,x')，从不构造高维特征 φ；
  - **支持向量稀疏**（KKT 互补松弛）：RBF 的支持向量数 ≪ n；
  - **核矩阵合法性**（Mercer）：RBF 核矩阵对称且半正定。

软间隔对偶用逐坐标投影上升求解（无偏置版），盒约束 0≤α≤C。
纯 NumPy、固定种子。
运行 `python -m ch17.kernels_and_svm` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np


def make_circles(rng, n, noise=0.08):
    n_out = n // 2
    n_in = n - n_out
    t_out = rng.uniform(0, 2 * np.pi, n_out)
    t_in = rng.uniform(0, 2 * np.pi, n_in)
    X_out = np.c_[np.cos(t_out), np.sin(t_out)] * 1.0
    X_in = np.c_[np.cos(t_in), np.sin(t_in)] * 0.4
    X = np.r_[X_out, X_in] + noise * rng.standard_normal((n, 2))
    y = np.r_[np.ones(n_out), -np.ones(n_in)]
    return X, y


def linear_kernel(A, B):
    return A @ B.T


def rbf_kernel(A, B, sigma=0.5):
    a2 = np.sum(A ** 2, 1)[:, None]
    b2 = np.sum(B ** 2, 1)[None, :]
    return np.exp(-(a2 + b2 - 2 * A @ B.T) / (2 * sigma ** 2))


def train_svm(K, y, C=1.0, iters=200):
    """软间隔对偶（无偏置）：逐坐标闭式最优 + 盒约束投影。"""
    n = len(y)
    Q = (y[:, None] * y[None, :]) * K
    a = np.zeros(n)
    diag = np.maximum(np.diag(Q), 1e-12)
    for _ in range(iters):
        for i in range(n):
            grad_i = 1.0 - (Q[i] @ a)
            a[i] = min(max(a[i] + grad_i / diag[i], 0.0), C)
    return a


def accuracy(a, y_tr, K_x_tr, y):
    f = K_x_tr @ (a * y_tr)
    return float(np.mean(np.sign(f) == y))


def run(seed=0, n=200, C=1.0):
    rng = np.random.default_rng(seed)
    X_tr, y_tr = make_circles(rng, n)
    X_te, y_te = make_circles(rng, n)

    a_lin = train_svm(linear_kernel(X_tr, X_tr), y_tr, C)
    acc_lin = accuracy(a_lin, y_tr, linear_kernel(X_te, X_tr), y_te)

    Ktr_rbf = rbf_kernel(X_tr, X_tr)
    a_rbf = train_svm(Ktr_rbf, y_tr, C)
    acc_rbf = accuracy(a_rbf, y_tr, rbf_kernel(X_te, X_tr), y_te)

    return {
        "n": n,
        "acc_linear": acc_lin,
        "sv_linear": int(np.sum(a_lin > 1e-5)),
        "acc_rbf": acc_rbf,
        "sv_rbf": int(np.sum(a_rbf > 1e-5)),
        "free_sv": int(np.sum((a_rbf > 1e-5) & (a_rbf < C - 1e-5))),
        "bounded_sv": int(np.sum(a_rbf >= C - 1e-5)),
        "K_sym_err": float(np.max(np.abs(Ktr_rbf - Ktr_rbf.T))),
        "K_min_eig": float(np.min(np.linalg.eigvalsh(Ktr_rbf))),
    }


def main() -> None:
    r = run()
    print(f"n_train = {r['n']}")
    print(f"[linear] test acc = {r['acc_linear']:.3f}   #SV = {r['sv_linear']}")
    print(f"[rbf]    test acc = {r['acc_rbf']:.3f}   #SV = {r['sv_rbf']}")
    print(f"[rbf] free SV (0<a<C) = {r['free_sv']}   bounded SV (a=C) = {r['bounded_sv']}")
    print(f"RBF K symmetric?  max|K - K^T| = {r['K_sym_err']:.1e}")
    print(f"RBF K min eigenvalue = {r['K_min_eig']:.2e}  ( >=0 => PSD )")


if __name__ == "__main__":
    main()
