"""The AI Mind · 第 16 章配套代码：线性模型

在一个易过拟合的问题（40 训练样本、20 特征、仅前 3 个有用）上跑正则化路径，
把三条性质跑成可核验的事实：
  (i)  岭回归 (ridge, L2)：||w||₂ 随 λ 单调下降却从不置零（只收缩）；
  (ii) lasso (L1)：出现精确的 0，非零个数阶梯式减少（稀疏/特征选择）；
  (iii) 存在 λ>0 使测试误差低于 λ=0（正则改善泛化）。

两个求解器都最小化不带 1/n 的目标（与正文公式同尺度）：
  ridge: ½‖Xw-y‖² + (λ/2)‖w‖²   （闭式解）
  lasso: ½‖Xw-y‖² + λ‖w‖₁        （坐标下降 + 软阈值）

纯 NumPy、固定种子。
运行 `python -m ch16.linear_models` 打印正文那两条正则化路径。
"""
from __future__ import annotations

import numpy as np


def make_problem(seed=0, n_train=40, n_test=400, d=20, k=3, sigma=0.5):
    rng = np.random.default_rng(seed)
    w_true = np.zeros(d)
    w_true[:k] = [3.0, -2.0, 1.5]

    def make_xy(n):
        X = rng.standard_normal((n, d))
        return X, X @ w_true + sigma * rng.standard_normal(n)

    X_tr, y_tr = make_xy(n_train)
    X_te, y_te = make_xy(n_test)
    mu, sd = X_tr.mean(0), X_tr.std(0)
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
    ym = y_tr.mean()
    return X_tr, y_tr - ym, X_te, y_te, ym, d


def ridge(X, y, lam, d):
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def _soft(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def lasso(X, y, lam, d, iters=1000):
    w = np.zeros(d)
    col_sq = (X ** 2).sum(0)
    for _ in range(iters):
        for j in range(d):
            rho = X[:, j] @ (y - X @ w + X[:, j] * w[j])
            w[j] = _soft(rho, lam) / col_sq[j]
    return w


def nnz(w, tol=1e-6):
    return int(np.sum(np.abs(w) > tol))


def ridge_path(lams=(0.0, 0.1, 0.3, 1.0, 10.0, 100.0)):
    X_tr, y_c, X_te, y_te, ym, d = make_problem()
    rows = []
    for lam in lams:
        w = ridge(X_tr, y_c, lam, d)
        test = float(np.mean((X_te @ w + ym - y_te) ** 2))
        rows.append((lam, float(np.linalg.norm(w)), nnz(w), test))
    return rows


def lasso_path(lams=(0.0, 1.0, 5.0, 20.0, 50.0)):
    X_tr, y_c, X_te, y_te, ym, d = make_problem()
    rows = []
    for lam in lams:
        w = ridge(X_tr, y_c, 1e-8, d) if lam == 0 else lasso(X_tr, y_c, lam, d)
        test = float(np.mean((X_te @ w + ym - y_te) ** 2))
        rows.append((lam, float(np.abs(w).sum()), nnz(w), test))
    return rows


def main() -> None:
    print("=== RIDGE (L2) ===")
    for lam, norm, k, test in ridge_path():
        print(f"lam={lam:6.1f}  ||w||2={norm:6.3f}  nnz={k:2d}  test={test:7.3f}")
    print("=== LASSO (L1) ===")
    for lam, norm, k, test in lasso_path():
        print(f"lam={lam:6.1f}  ||w||1={norm:6.3f}  nnz={k:2d}  test={test:7.3f}")


if __name__ == "__main__":
    main()
