"""The AI Mind · 第 15 章配套代码：学习理论——泛化为何可能

让泛化理论的三条断言现形（数据分布 D 已知，故可精确逼近真实风险 R）：
  一. 一致收敛：最大偏差 sup_h|R̂(h)-R(h)| 随 n 以 1/√n 收缩，
      且被有限类泛化界 sqrt((lnM+ln(2/δ))/(2n)) 从上方罩住（Hoeffding+union bound）。
  二. 泛化差距随假设类容量增大而扩大（额外特征 train/test 同分布，差距来自真过拟合）。
  三. 随机标签：高容量模型把训练误差压到 0（纯记忆），测试误差≈随机（Zhang 2017 精神）。

纯 NumPy、每个演示用自己固定种子的局部 RNG。
运行 `python -m ch15.learning_theory` 打印正文那三组可核验输出。
"""
from __future__ import annotations

import numpy as np

D_DIM = 20
W_STAR = np.random.default_rng(0).standard_normal(D_DIM)
LABEL_NOISE = 0.05


def sample(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D_DIM))
    y = (X @ W_STAR > 0).astype(np.float64)
    flip = rng.random(n) < LABEL_NOISE
    y[flip] = 1.0 - y[flip]
    return X, y


def fit_erm(X, y, n_feat, ridge=1e-6):
    Phi = np.concatenate([X[:, :n_feat], np.ones((len(X), 1))], axis=1)
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ (2 * y - 1))


def err(X, y, w, n_feat):
    Phi = np.concatenate([X[:, :n_feat], np.ones((len(X), 1))], axis=1)
    return float(np.mean(((Phi @ w > 0).astype(np.float64)) != y))


def uniform_convergence(X_test, y_test, M=1000, delta=0.05, ns=(50, 100, 200, 400, 800, 1600, 3200)):
    """演示一：返回 [(n, max_dev, max_dev*sqrt(n), theoretical_bound)]。"""
    rng_H = np.random.default_rng(1)
    Wc = rng_H.standard_normal((M, D_DIM))
    Bc = rng_H.standard_normal(M)
    Ptest = (X_test @ Wc.T + Bc > 0).astype(np.float64)
    R_all = np.mean(Ptest != y_test[:, None], axis=0)
    rows = []
    for n in ns:
        devs = []
        for s in range(40):
            X, y = sample(n, seed=n * 1000 + s)
            Rhat_all = np.mean(((X @ Wc.T + Bc > 0).astype(np.float64)) != y[:, None], axis=0)
            devs.append(np.max(np.abs(Rhat_all - R_all)))
        md = float(np.mean(devs))
        bound = float(np.sqrt((np.log(M) + np.log(2 / delta)) / (2 * n)))
        rows.append((n, md, md * np.sqrt(n), bound))
    return rows


def capacity_gap(X_test, y_test, n=120, nfeats=(2, 5, 10, 20, 40, 80, 119)):
    """演示二：返回 [(n_feat, R_hat, R, gap)]。"""
    Xf, yf = sample(n, seed=7)
    rng2 = np.random.default_rng(7)
    rows = []
    for nf in nfeats:
        extra = max(0, nf - D_DIM)
        if extra > 0:
            Xf2 = np.concatenate([Xf, rng2.standard_normal((n, extra))], axis=1)
            Xt2 = np.concatenate([X_test, rng2.standard_normal((len(X_test), extra))], axis=1)
        else:
            Xf2, Xt2 = Xf, X_test
        w = fit_erm(Xf2, yf, n_feat=nf)
        Rhat, R = err(Xf2, yf, w, nf), err(Xt2, y_test, w, nf)
        rows.append((nf, Rhat, R, R - Rhat))
    return rows


def random_label_memorization(X_test, n=120, hidden=300):
    """演示三：返回 (train_err, test_err) 在随机标签下。"""
    Xr, _ = sample(n, seed=3)
    rng3 = np.random.default_rng(42)
    y_rand = (rng3.random(n) > 0.5).astype(np.float64)
    P = rng3.standard_normal((D_DIM, hidden))
    Feat = np.tanh(Xr @ P)
    w = np.linalg.solve(Feat.T @ Feat + 1e-8 * np.eye(hidden), Feat.T @ (2 * y_rand - 1))
    train_err = float(np.mean(((Feat @ w > 0).astype(np.float64)) != y_rand))
    Feat_test = np.tanh(X_test[:2000] @ P)
    y_rand_test = (rng3.random(2000) > 0.5).astype(np.float64)
    test_err = float(np.mean(((Feat_test @ w > 0).astype(np.float64)) != y_rand_test))
    return train_err, test_err


def run():
    X_test, y_test = sample(50000, seed=999)
    return {
        "uniform": uniform_convergence(X_test, y_test),
        "capacity": capacity_gap(X_test, y_test),
        "random_labels": random_label_memorization(X_test),
    }


def main() -> None:
    res = run()
    print("演示一：一致收敛 —— 最大偏差 sup_h|R_hat-R| 随 n 以 1/sqrt(n) 收缩")
    print(f"{'n':>7} {'max_dev':>9} {'max_dev*sqrt(n)':>16} {'理论界':>9}")
    for n, md, mds, bound in res["uniform"]:
        print(f"{n:>7} {md:>9.4f} {mds:>16.3f} {bound:>9.4f}")
    print("\n演示二：泛化差距随容量扩大 (固定 n=120)")
    print(f"{'n_feat':>7} {'R_hat':>8} {'R':>8} {'gap':>8}")
    for nf, rhat, r, gap in res["capacity"]:
        print(f"{nf:>7} {rhat:>8.4f} {r:>8.4f} {gap:>8.4f}")
    tr, te = res["random_labels"]
    print("\n演示三：随机标签 —— 记忆的极端")
    print(f"训练误差(随机标签): {tr:.4f}\n测试误差(随机标签): {te:.4f}")


if __name__ == "__main__":
    main()
