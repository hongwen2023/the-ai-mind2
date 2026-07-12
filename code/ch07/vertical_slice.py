"""The AI Mind · 第 7 章配套代码：垂直切片——先跑通一个会学习的系统

Book I 收官顶点：从零实现 逻辑回归 (logistic regression) + 梯度下降 (gradient descent)，
把前六章零件拼成一个真正"通过优化从数据学出参数"的学习循环：
  数据(Ch5) → 划分(Ch6) → 标准化(仅用训练集统计量, Ch4/Ch6) → 模型 H=线性+sigmoid(Ch1)
  → 交叉熵损失 = R̂(Ch1) → 梯度下降最小化 R̂(Ch1 ERM) → 留出集评估(Ch6)

含**梯度检验 (gradient check)**：有限差分数值梯度对照解析梯度（前指 Ch23 micro-autograd）。

数据用 Cholesky 生成相关高斯（而非 multivariate_normal，后者默认 SVD 分解跨 LAPACK
构建不保证逐位一致），以保证跨机器可复现。

运行 `python -m ch07.vertical_slice` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np


def make_data(rng, n=400):
    """两团近似线性可分的相关高斯。Cholesky 分解保证跨机器逐位一致。"""
    n_half = n // 2
    mean0 = np.array([-1.5, -1.5])
    mean1 = np.array([1.5, 1.5])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(cov)
    X0 = mean0 + rng.standard_normal((n_half, 2)) @ L.T
    X1 = mean1 + rng.standard_normal((n_half, 2)) @ L.T
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_half), np.ones(n_half)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def loss(X, y, w, b):
    """交叉熵 = Ch1 的经验风险 R̂。"""
    p = predict_proba(X, w, b)
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def grad(X, y, w, b):
    """解析梯度：平均(预测误差 × 输入)。"""
    err = predict_proba(X, w, b) - y
    return X.T @ err / len(y), np.mean(err)


def accuracy(X, y, w, b):
    return float(np.mean((predict_proba(X, w, b) >= 0.5) == (y == 1)))


def gradient_check(X, y, w, b, eps=1e-6):
    """有限差分 vs 解析梯度，返回最大相对误差。"""
    gw, gb = grad(X, y, w, b)
    num_gw = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy(); wp[i] += eps
        wm = w.copy(); wm[i] -= eps
        num_gw[i] = (loss(X, y, wp, b) - loss(X, y, wm, b)) / (2 * eps)
    num_gb = (loss(X, y, w, b + eps) - loss(X, y, w, b - eps)) / (2 * eps)
    ana = np.append(gw, gb)
    num = np.append(num_gw, num_gb)
    rel = np.abs(num - ana) / (np.abs(num) + np.abs(ana) + 1e-12)
    return float(rel.max())


def run_experiment(seed: int = 7, eta: float = 0.5, iters: int = 2000):
    """复现正文那台学习机的完整实验，返回关键结果 dict。"""
    rng = np.random.default_rng(seed)
    X, y = make_data(rng)
    n = X.shape[0]
    n_train = int(0.7 * n)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    # 标准化：划分之后，只用训练集统计量（否则测试信息泄漏）
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    grad_rel = gradient_check(Xtr, ytr, rng.normal(size=2), 0.0)

    w, b = np.zeros(2), 0.0
    init_loss = loss(Xtr, ytr, w, b)
    for _ in range(iters):
        gw, gb = grad(Xtr, ytr, w, b)
        w -= eta * gw
        b -= eta * gb
    return {
        "grad_check_rel": grad_rel,
        "init_loss": init_loss,
        "final_loss": loss(Xtr, ytr, w, b),
        "train_acc": accuracy(Xtr, ytr, w, b),
        "test_acc": accuracy(Xte, yte, w, b),
        "w": w, "b": b,
    }


def main() -> None:
    r = run_experiment()
    print(f"梯度检验 最大相对误差: {r['grad_check_rel']:.2e}")
    print(f"初始损失: {r['init_loss']:.4f}   最终损失: {r['final_loss']:.4f}")
    print(f"最终训练准确率: {r['train_acc']:.3f}   最终测试准确率: {r['test_acc']:.3f}")
    print(f"最终参数 w=({r['w'][0]:.2f}, {r['w'][1]:.2f})   b={r['b']:.2f}")


if __name__ == "__main__":
    main()
