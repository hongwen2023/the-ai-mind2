"""The AI Mind · 第 6 章配套代码：从样本到规律——归纳第一课

用 k 近邻 (k-nearest-neighbors, kNN) 把"记忆冒充理解"演示到赤裸：
kNN 字面上把训练集存下来，k 就是"奥卡姆剃刀"的旋钮。

关键现象（确定性，seed=6，PCG64/default_rng）：
  - k=1 训练误差恒为 0——每个训练点的最近邻是它自己，纯记忆不是学习；
  - 存在某个 k>1，其测试误差明显低于 k=1（背得最熟 ≠ 泛化最好）；
  - 测试误差随 k 先降后升（U 型），与 Ch1 多项式次数 d 那条 U 型是同一个现象。

运行 `python -m ch06.induction_first_lesson` 打印正文那张训练/测试误差表。
"""
from __future__ import annotations

import numpy as np

KS = [1, 3, 5, 9, 15, 25, 51, 101, 201, 401]


def make_two_moons(rng, n_per_class, noise):
    """两个交错半月形，非线性可分。加噪制造无法被完美学出的抖动。"""
    t = rng.uniform(0.0, np.pi, size=n_per_class)
    x1 = np.stack([np.cos(t), np.sin(t)], axis=1)
    x2 = np.stack([1.0 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([x1, x2], axis=0)
    X = X + rng.normal(0.0, noise, size=X.shape)
    y = np.concatenate([np.zeros(n_per_class, dtype=int),
                        np.ones(n_per_class, dtype=int)])
    return X, y


def knn_predict(Xq, Xref, yref, k):
    """对每个查询点取最近 k 个邻居投票。k 取奇数避免二分类平票。"""
    d = np.sqrt(((Xq[:, None, :] - Xref[None, :, :]) ** 2).sum(axis=2))
    idx = np.argsort(d, axis=1)[:, :k]
    votes = yref[idx]
    return (votes.mean(axis=1) >= 0.5).astype(int)


def err(pred, y):
    return float(np.mean(pred != y))


def run_table(seed: int = 6, n_train: int = 800):
    """复现正文那张 (k, train_err, test_err) 表。返回 list[(k, train_err, test_err)]。"""
    rng = np.random.default_rng(seed)
    X, y = make_two_moons(rng, n_per_class=800, noise=0.25)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    rows = []
    for k in KS:
        ptr = knn_predict(Xtr, Xtr, ytr, k)
        pte = knn_predict(Xte, Xtr, ytr, k)
        rows.append((k, err(ptr, ytr), err(pte, yte)))
    return rows


def main() -> None:
    print(f"{'k':>4} {'train_err':>10} {'test_err':>10}")
    for k, tr, te in run_table():
        print(f"{k:>4} {tr:>10.3f} {te:>10.3f}")


if __name__ == "__main__":
    main()
