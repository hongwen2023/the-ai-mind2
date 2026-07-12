"""The AI Mind · 第 20 章配套代码：隐变量与 EM——高斯混合

在三个重叠、椭圆状（带旋转、不等方差）的高斯簇——k-means 会失手的场景——上
从零实现 GMM-EM，验证：
  - EM 产生一列**单调不减的对数似然**（ELBO 坐标上升的可见证据）；
  - 责任（软归属）每行和为 1，边界点对多个成分责任都显著非零（≠ k-means 硬分配）；
  - 恢复出接近真值的成分参数。

数值稳定：责任在对数域用 log-sum-exp（回指 Ch9），M 步给 Σ 加 1e-6·I
作为防成分坍缩/似然奇异发散的护栏（见数学线）。
纯 NumPy、固定种子。
运行 `python -m ch20.gmm_em` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np


def log_gauss(X, mu, Sigma, D):
    L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(D))
    diff = X - mu
    sol = np.linalg.solve(L, diff.T).T
    quad = np.sum(sol ** 2, axis=1)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2 * np.pi) + logdet + quad)


def logsumexp(A, axis):
    m = np.max(A, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))).squeeze(axis)


def make_data(seed=20):
    rng = np.random.default_rng(seed)
    mu_true = np.array([[0., 0.], [4., 4.], [6., 0.]])
    covs = [np.array([[1.5, 0.9], [0.9, 1.0]]),
            np.array([[1.0, -0.7], [-0.7, 1.2]]),
            np.array([[0.6, 0.0], [0.0, 2.5]])]
    pis = [0.40, 0.35, 0.25]
    X = np.vstack([rng.multivariate_normal(mu_true[k], covs[k], size=int(200 * pis[k]))
                   for k in range(3)])
    return X, mu_true


def run(seed=20, iters=40):
    X, mu_true = make_data(seed)
    N, D = X.shape
    K = 3
    mu = np.array([[1., 1.], [3., 3.], [5., 1.]])
    Sigma = np.array([np.eye(2) for _ in range(K)])
    pi = np.ones(K) / K
    lls = []
    gamma = None
    for _ in range(iters):
        logw = np.log(pi)[None, :] + np.column_stack(
            [log_gauss(X, mu[k], Sigma[k], D) for k in range(K)])
        lls.append(float(np.sum(logsumexp(logw, axis=1))))
        gamma = np.exp(logw - logsumexp(logw, axis=1)[:, None])
        Nk = gamma.sum(axis=0)
        pi = Nk / N
        mu = (gamma.T @ X) / Nk[:, None]
        for k in range(K):
            diff = X - mu[k]
            Sigma[k] = (gamma[:, k, None] * diff).T @ diff / Nk[k] + 1e-6 * np.eye(D)
    ent = -np.sum(gamma * np.log(gamma + 1e-12), axis=1)
    boundary = [(X[j], gamma[j]) for j in np.argsort(-ent)[:3]]
    return {
        "lls": np.array(lls),
        "resp_rowsum_err": float(np.max(np.abs(gamma.sum(1) - 1))),
        "pi": pi, "mu": mu, "mu_true": mu_true,
        "boundary": boundary,
        "max_resp_of_top_boundary": float(np.max(boundary[0][1])),
    }


def main() -> None:
    r = run()
    print("前 8 轮对数似然:", np.round(r["lls"][:8], 3))
    print("收敛值:", round(r["lls"][-1], 3),
          "| 单调不减:", bool(np.all(np.diff(r["lls"]) >= -1e-6)))
    print("责任每行和偏离 1 的最大值:", r["resp_rowsum_err"])
    print("恢复的 pi:", np.round(r["pi"], 3))
    print("恢复的 mu:\n", np.round(r["mu"], 2))
    for x, g in r["boundary"]:
        print(f"边界点 x={np.round(x, 2)}  责任={np.round(g, 3)}")


if __name__ == "__main__":
    main()
