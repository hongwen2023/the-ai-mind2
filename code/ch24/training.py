"""The AI Mind · 第 24 章配套代码：训练的科学与艺术

一个小型向量化 MLP（two-moons 上），把"让训练 work"的技巧跑成可核验的事实：
  一. **初始化的生死**：全零初始化因对称性学不动（隐藏层各神经元恒等、梯度相同，
      loss 停在 ln2≈0.693、测试准确率≈随机 0.49）；He 初始化训练到高准确率。
  二. **优化器对照**：全批 GD vs Adam，同样步数下 Adam 到更低的损失。

含稳定 softmax/交叉熵（log-sum-exp，回指 Ch9）、权重衰减（L2，只作用于权重）。
纯 NumPy、固定种子。
运行 `python -m ch24.training` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np

SIZES = [2, 16, 16, 2]


def make_moons(n=400, noise=0.20, seed=1):
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out
    t_out = np.pi * rng.random(n_out)
    outer = np.stack([np.cos(t_out), np.sin(t_out)], 1)
    t_in = np.pi * rng.random(n_in)
    inner = np.stack([1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5], 1)
    X = np.concatenate([outer, inner], 0)
    y = np.concatenate([np.zeros(n_out), np.ones(n_in)]).astype(int)
    X += noise * rng.standard_normal(X.shape)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _data():
    X, y = make_moons()
    X = (X - X.mean(0)) / X.std(0)
    return X[:300], y[:300], X[300:], y[300:]


def init(mode, sizes, seed=0):
    rng = np.random.default_rng(seed)
    P = {}
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        if mode == "zero":
            W = np.zeros((fan_in, fan_out))
        elif mode == "xavier":
            W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(1.0 / fan_in)
        elif mode == "he":
            W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        P[f"W{i}"], P[f"b{i}"] = W, np.zeros(fan_out)
    return P


def forward(P, X, L):
    cache = {"a0": X}; a = X
    for i in range(L):
        z = a @ P[f"W{i}"] + P[f"b{i}"]
        cache[f"z{i}"] = z
        a = np.maximum(0, z) if i < L - 1 else z
        cache[f"a{i+1}"] = a
    return a, cache


def softmax_ce(logits, y):
    m = logits.max(1, keepdims=True)
    logZ = m[:, 0] + np.log(np.exp(logits - m).sum(1))
    logp = logits - logZ[:, None]
    n = logits.shape[0]
    loss = -logp[np.arange(n), y].mean()
    g = np.exp(logp); g[np.arange(n), y] -= 1; g /= n
    return loss, g


def backward(P, cache, dlogits, L):
    G = {}; dz = dlogits
    for i in reversed(range(L)):
        G[f"W{i}"] = cache[f"a{i}"].T @ dz
        G[f"b{i}"] = dz.sum(0)
        da = dz @ P[f"W{i}"].T
        if i > 0:
            dz = da * (cache[f"z{i-1}"] > 0)
    return G


def acc(P, X, y, L):
    return float((forward(P, X, L)[0].argmax(1) == y).mean())


def train(Xtr, ytr, mode, opt, steps=400, lr=0.1, wd=0.0, seed=0):
    L = len(SIZES) - 1
    P = init(mode, SIZES, seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    losses = []
    for t in range(1, steps + 1):
        logits, cache = forward(P, Xtr, L)
        loss, dlogits = softmax_ce(logits, ytr)
        losses.append(loss)
        G = backward(P, cache, dlogits, L)
        for k in P:
            g = G[k] + (wd * P[k] if k.startswith("W") else 0.0)
            if opt == "gd":
                P[k] -= lr * g
            else:
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g * g
                mhat = m[k] / (1 - b1 ** t)
                vhat = v[k] / (1 - b2 ** t)
                P[k] -= lr * mhat / (np.sqrt(vhat) + eps)
    return P, losses, L


def run():
    Xtr, ytr, Xte, yte = _data()
    P0, l0, L = train(Xtr, ytr, "zero", "gd")
    Ph, lh, _ = train(Xtr, ytr, "he", "gd")
    _, c0 = forward(P0, Xtr, L)
    Ps, ls, _ = train(Xtr, ytr, "he", "gd", lr=0.1)
    Pa, la, _ = train(Xtr, ytr, "he", "adam", lr=0.01)
    return {
        "zero_loss": l0[-1], "zero_acc": acc(P0, Xte, yte, L),
        "zero_hidden_std": float(np.max(c0["a1"].std(0))),
        "he_loss": lh[-1], "he_acc": acc(Ph, Xte, yte, L),
        "gd_final": ls[-1], "adam_final": la[-1],
        "gd_acc": acc(Ps, Xte, yte, L), "adam_acc": acc(Pa, Xte, yte, L),
    }


def main() -> None:
    r = run()
    print("[演示一] 全零初始化: train loss %.4f, test acc %.3f  (隐藏层各列 std max=%.4g，对称)"
          % (r["zero_loss"], r["zero_acc"], r["zero_hidden_std"]))
    print("[演示一] He  初始化: train loss %.4f, test acc %.3f" % (r["he_loss"], r["he_acc"]))
    print("[演示二] 末端: GD %.4f (acc %.3f) | Adam %.4f (acc %.3f)"
          % (r["gd_final"], r["gd_acc"], r["adam_final"], r["adam_acc"]))


if __name__ == "__main__":
    main()
