"""The AI Mind · 第 19 章配套代码：无监督学习——发现结构

两个演示：
  一. k-means (Lloyd)：验证簇内平方和 J 单调不增并收敛（块坐标下降，回指 Ch14）。
  二. PCA via SVD：验证 PCA = 中心化数据的 SVD，
      且降到 k 维的重构误差 = 丢弃奇异值的平方和（Eckart–Young，兑现 Ch8）。

纯 NumPy、固定种子。
运行 `python -m ch19.unsupervised` 打印正文那两组可核验输出。
"""
from __future__ import annotations

import numpy as np


# ---------- k-means ----------
def make_blobs(rng, n_per=200):
    centers = np.array([[0.0, 0.0], [4.0, 4.0], [-4.0, 4.0]])
    return np.vstack([c + rng.normal(scale=1.0, size=(n_per, 2)) for c in centers])


def wcss(X, labels, mu):
    return float(np.sum((X - mu[labels]) ** 2))


def kmeans_lloyd(X, k, rng, n_iter=50):
    n = X.shape[0]
    mu = X[rng.choice(n, size=k, replace=False)].copy()
    history = []
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        history.append(wcss(X, labels, mu))
        new_mu = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else mu[j]
                           for j in range(k)])
        history.append(wcss(X, labels, new_mu))
        if np.allclose(new_mu, mu):
            mu = new_mu
            break
        mu = new_mu
    return labels, mu, history


def run_kmeans(seed=0):
    X = make_blobs(np.random.default_rng(seed))
    labels, mu, history = kmeans_lloyd(X, k=3, rng=np.random.default_rng(seed))
    return history


# ---------- PCA via SVD ----------
def run_pca(seed=7, n=500, d=20, r=2, k=2):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, r)) * np.array([5.0, 2.0])
    B = rng.normal(size=(r, d))
    X = Z @ B + rng.normal(scale=0.3, size=(n, d))
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    explained = S ** 2 / np.sum(S ** 2)
    Xk = U[:, :k] * S[:k] @ Vt[:k]           # 秩 k 最佳逼近
    recon_err = float(np.sum((Xc - Xk) ** 2))
    tail = float(np.sum(S[k:] ** 2))          # 丢弃奇异值平方和
    return {"explained": explained, "top_k_ratio": float(explained[:k].sum()),
            "recon_err": recon_err, "dropped_sv2": tail}


def main() -> None:
    history = run_kmeans()
    print("k-means J 序列（每次迭代含 分配后 / 更新后 两个值）:")
    print(np.round(history, 2))
    print("单调不增?", all(a >= b - 1e-9 for a, b in zip(history, history[1:])))
    print("最终 J = %.4f, 迭代次数 = %d" % (history[-1], len(history) // 2))

    p = run_pca()
    print("\n前 5 个解释方差比:", np.round(p["explained"][:5], 4))
    print("前 2 个主成分累计解释方差比:", round(p["top_k_ratio"], 4))
    print("重构误差            = %.4f" % p["recon_err"])
    print("丢弃奇异值平方和    = %.4f" % p["dropped_sv2"])
    print("两者相符（容差内）? ", np.isclose(p["recon_err"], p["dropped_sv2"], rtol=1e-6))


if __name__ == "__main__":
    main()
