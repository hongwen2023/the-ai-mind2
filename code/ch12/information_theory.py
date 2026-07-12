"""The AI Mind · 第 12 章配套代码：信息论——惊奇、熵与压缩

把命名定理算成可打印的事实：
  - 恒等式 H(p,q) = H(p) + D(p‖q)，且 D(p‖q) ≥ 0（=0 当且仅当 p=q）；
  - KL 不对称：D(p‖q) ≠ D(q‖p)；
  - **命名定理**：固定数据分布 p，交叉熵 H(p,q) 在 q=p 处触底、且最小值 = H(p)
    （此时 KL=0）——即"最小化交叉熵 = 让 q 逼近 p"；
  - 压缩直觉：dyadic 分布下按 -log2 p 配码长，平均码长 = 熵。

以 bit 计（log2）。纯 NumPy、确定性。
运行 `python -m ch12.information_theory` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np


def entropy(p):
    p = np.asarray(p, dtype=float)
    nz = p > 0                       # 0·log0 约定为 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(q[nz])))


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / q[nz])))


def scan_cross_entropy(p, steps=11):
    """沿 q_t=(1-t)p + t·uniform 扫描，返回 [(t, H(p,q_t), D(p‖q_t))]。t=0 即 q=p。"""
    p = np.asarray(p, float)
    u = np.ones(len(p)) / len(p)
    rows = []
    for t in np.linspace(0.0, 1.0, steps):
        qt = (1 - t) * p + t * u
        rows.append((float(t), cross_entropy(p, qt), kl(p, qt)))
    return rows


def main() -> None:
    p = np.array([0.5, 0.25, 0.125, 0.125])   # dyadic
    q = np.array([0.25, 0.25, 0.25, 0.25])    # 均匀：一个"用错的密码本"
    Hp, Hpq, Dpq = entropy(p), cross_entropy(p, q), kl(p, q)
    print(f"H(p)      = {Hp:.4f} bits")
    print(f"H(p,q)    = {Hpq:.4f} bits")
    print(f"D(p||q)   = {Dpq:.4f} bits")
    print(f"恒等式核验 H(p)+D = {Hp + Dpq:.4f}  vs  H(p,q) = {Hpq:.4f}")
    print(f"D(p||p)   = {kl(p, p):.4f}  (应为 0)")

    q2 = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"\n[不对称] D(p||q2) = {kl(p, q2):.4f}   D(q2||p) = {kl(q2, p):.4f}   (两者不等)")

    print("\n  t     H(p, q_t)      D(p||q_t)")
    best_t, best_ce = None, np.inf
    for t, ce, d in scan_cross_entropy(p):
        print(f"{t:0.1f}    {ce:0.4f}       {d:0.4f}")
        if ce < best_ce:
            best_ce, best_t = ce, t
    print(f"交叉熵最小在 t={best_t:.1f}（即 q=p），最小值={best_ce:.4f} = H(p)={Hp:.4f}")

    code_len = -np.log2(p)
    avg = float(np.sum(p * code_len))
    print(f"\n最优码长 = {code_len.astype(int)},  平均码长 = {avg:.4f} = H(p) = {Hp:.4f}")


if __name__ == "__main__":
    main()
