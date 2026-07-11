"""The AI Mind · 第 14 章配套代码：优化——如何下山

偿还 Ch7 第一张欠条（GD 为何收敛），把三件事跑成可核验的事实：
  一/二. 凸二次目标上的 GD 步长三态：η 太小(收敛慢) / η=1/L(合适) / η>2/L(发散)。
         光滑常数 L = A 的最大特征值（回指 Ch8 谱）。
  三.   SGD vs 全批 GD：全批收敛到机器精度；固定步长 SGD 停在噪声球里；
         衰减步长（Robbins-Monro 精神）把噪声球压下去。

纯 NumPy、固定种子（legacy seed，与正文数值一致）。
运行 `python -m ch14.optimization` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np


def run(seed: int = 0):
    """复现正文全部数值，返回结果 dict。"""
    np.random.seed(seed)

    # ---- 凸二次目标 f(w)=½wᵀAw-bᵀw ----
    d = 20
    Q, _ = np.linalg.qr(np.random.randn(d, d))
    eigs = np.linspace(1.0, 30.0, d)
    A = (Q * eigs) @ Q.T
    A = 0.5 * (A + A.T)
    b = np.random.randn(d)
    w_star = np.linalg.solve(A, b)
    f_star = 0.5 * w_star @ A @ w_star - b @ w_star
    L, mu = float(np.max(eigs)), float(np.min(eigs))

    def f(w):
        return 0.5 * w @ A @ w - b @ w

    def grad(w):
        return A @ w - b

    def run_gd(eta, steps=200):
        w = np.zeros(d)
        hist = []
        for _ in range(steps):
            w = w - eta * grad(w)
            hist.append((f(w) - f_star, np.linalg.norm(w - w_star)))
        return np.array(hist)

    three_states = {}
    for eta, tag in [(0.02 / L, "small"), (1.0 / L, "good"), (2.05 / L, "large")]:
        h = run_gd(eta)
        gap = h[-1, 0]
        diverged = (not np.isfinite(gap)) or (gap > h[0, 0])
        three_states[tag] = {"eta": eta, "final_gap": float(gap), "diverged": bool(diverged)}

    # ---- SGD vs 全批 GD（线性回归平方损失）----
    n, p = 2000, 10
    X = np.random.randn(n, p)
    w_true = np.random.randn(p)
    y = X @ w_true + 0.1 * np.random.randn(n)
    w_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    R_star = float(np.mean((X @ w_ls - y) ** 2))

    def R(w):
        return float(np.mean((X @ w - y) ** 2))

    Lls = 2.0 * np.max(np.linalg.eigvalsh(X.T @ X)) / n
    w = np.zeros(p)
    for _ in range(300):
        w = w - (1.0 / Lls) * (2.0 / n) * X.T @ (X @ w - y)
    full_gd_gap = R(w) - R_star

    batch = 16
    # 固定步长 SGD
    w = np.zeros(p); eta_c = 0.1
    for _ in range(30):
        perm = np.random.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            g = (2.0 / len(idx)) * X[idx].T @ (X[idx] @ w - y[idx])
            w = w - eta_c * g
    sgd_fixed_gap = R(w) - R_star

    # 衰减步长 SGD
    w = np.zeros(p); eta0 = 0.1; gstep = 0
    for _ in range(30):
        perm = np.random.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            g = (2.0 / len(idx)) * X[idx].T @ (X[idx] @ w - y[idx])
            gstep += 1
            w = w - (eta0 / (1.0 + 0.003 * gstep)) * g
    sgd_decay_gap = R(w) - R_star

    return {
        "L": L, "mu": mu, "f_star": f_star,
        "three_states": three_states,
        "gd_good_final": three_states["good"]["final_gap"],
        "full_gd_gap": full_gd_gap,
        "sgd_fixed_gap": sgd_fixed_gap,
        "sgd_decay_gap": sgd_decay_gap,
    }


def main() -> None:
    r = run()
    print(f"L = {r['L']:.1f}, mu = {r['mu']:.1f}, 1/L = {1/r['L']:.4f}, f_star = {r['f_star']:.6f}")
    print("\n[步长三态] 1/L = %.4f:" % (1 / r["L"]))
    for tag in ("small", "good", "large"):
        s = r["three_states"][tag]
        print(f"  eta={s['eta']:.4f} ({tag}): 末端 f-f* = {s['final_gap']:.3e}  "
              f"[{'发散' if s['diverged'] else '收敛'}]")
    print(f"\n全批 GD 末端 R-R* = {r['full_gd_gap']:.3e}")
    print(f"固定步长 SGD 末端 R-R* = {r['sgd_fixed_gap']:.3e}")
    print(f"衰减步长 SGD 末端 R-R* = {r['sgd_decay_gap']:.3e}")


if __name__ == "__main__":
    main()
