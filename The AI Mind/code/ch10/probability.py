"""The AI Mind · 第 10 章配套代码：概率论——不确定性的演算

让大数定律与集中在屏幕上发生：
  A. 蒙特卡洛 (Monte Carlo) 估计——样本均值随 n 增大收敛到真值，
     误差贴合 sigma/sqrt(n)（大数定律 + 1/√n 标度）。估偏币 p 与投点估 π。
  B. 经验验证 Hoeffding——重复实验统计"样本均值偏离期望 ≥ t"的经验尾频率，
     核对它确实 ≤ Hoeffding 上界 2·exp(-2·n·t²)。

纯 NumPy、固定种子。
运行 `python -m ch10.probability` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np


def estimate_coin(p_true=0.37, ns=(10, 100, 1000, 10000, 100000, 1000000), seed=42):
    """返回 [(n, est, |err|, sigma/sqrt(n))]。"""
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        samples = (rng.random(n) < p_true).astype(np.float64)
        est = float(samples.mean())
        sd = float(np.sqrt(p_true * (1 - p_true) / n))
        rows.append((n, est, abs(est - p_true), sd))
    return rows


def estimate_pi(ns=(10, 100, 1000, 10000, 100000, 1000000), seed=42):
    """投点估 π，返回 [(n, pi_est, |err|)]。"""
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        pts = rng.random((n, 2))
        inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2 <= 1.0)
        pi_est = 4.0 * float(inside.mean())
        rows.append((n, pi_est, abs(pi_est - np.pi)))
    return rows


def hoeffding_check(p=0.5, t=0.1, trials=200000, ns=(50, 100, 200, 400), seed=0):
    """返回 [(n, 经验尾频率, Hoeffding上界)]。"""
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        means = (rng.random((trials, n)) < p).mean(axis=1)
        emp = float(np.mean(np.abs(means - p) >= t))
        bound = 2.0 * np.exp(-2.0 * n * t ** 2)
        rows.append((n, emp, bound))
    return rows


def main() -> None:
    print("=== 演示 1a：估计偏币 p = 0.37 ===")
    for n, est, err, sd in estimate_coin():
        print(f"n={n:>8d}  est={est:.5f}  |err|={err:.5f}  sigma/sqrt(n)={sd:.5f}")
    print("\n=== 演示 1b：投点估计 pi ===")
    for n, pi_est, err in estimate_pi():
        print(f"n={n:>8d}  pi_est={pi_est:.5f}  |err|={err:.5f}")
    print("\n=== 演示 2：经验尾频率 vs Hoeffding 上界 2exp(-2 n t^2) ===")
    for n, emp, bound in hoeffding_check():
        print(f"n={n:>4d}  emp={emp:.5f}  bound={bound:.5f}  emp<=bound? {emp <= bound}")


if __name__ == "__main__":
    main()
