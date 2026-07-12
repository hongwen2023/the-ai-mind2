"""The AI Mind · 第 28 章配套代码：从零复现双下降（研究里程碑①）

把定义了这个领域的一张图——**双下降 (double descent)**——用纯 NumPy 从零复现。
配方是已知稳健触发双下降的组合：**随机特征 (random features) + 标签噪声 +
最小范数 (ridgeless) 解**。

核心事实（seed=0，确定性）：
  1. 测试风险在插值阈值 p=n=40 处冲上尖峰（≈13.1，σ_min→0 导致解范数/方差爆炸）；
  2. 过参数区 (p>>n) 第二次下降到 ≈0.40，**低于**经典甜蜜点 ≈0.82——双下降的强形式；
  3. 尖峰源于 σ_min→0 放大**一切**拟合误差（标签噪声 + tanh 特征对线性教师的
     近似误差/误设），噪声是加剧者而非唯一成因（noise=0 时尖峰仍在，≈6.6）；
  4. 玩具规模律：良设线性回归超额风险随 n 呈幂律，log-log 斜率 ≈ −0.95（理论 −1）。

连回概念：pinv=伪逆=最小范数解 (Ch8 SVD, X+ = VΣ+Uᵀ)；最小范数=隐式岭正则
(Ch16, λ→0)；整条曲线左半 = Ch11 偏差-方差；随机标签记忆证伪 Ch15 泛化界。

运行 `python -m ch28.science_of_dl` 打印数值表 + ASCII 双下降图。
"""
from __future__ import annotations

import numpy as np


def run(seed=0):
    rng = np.random.default_rng(seed)
    d = 20            # 潜在输入维度
    n = 40            # 训练样本数 —— 插值阈值出现在 p ≈ n
    n_test = 2000     # 独立测试集
    noise = 0.5       # 标签噪声（加剧尖峰，但非尖峰唯一成因）

    w_star = rng.standard_normal(d) / np.sqrt(d)          # 固定的线性教师函数

    def sample(m):
        X = rng.standard_normal((m, d))
        y = X @ w_star + noise * rng.standard_normal(m)
        return X, y

    Xtr, ytr = sample(n)
    Xte, yte = sample(n_test)

    Wfull = rng.standard_normal((d, 2000))                # 固定随机投影，按列嵌套

    def phi(X, p):
        # tanh 特征无法精确表示线性教师 —— 存在“近似误差(误设)”
        return np.tanh(X @ Wfull[:, :p])

    ps = [2, 5, 10, 20, 30, 38, 40, 42, 50, 70, 120, 300, 800, 2000]
    curve = []
    betas = {}
    for p in ps:
        Ptr = phi(Xtr, p)
        beta = np.linalg.pinv(Ptr) @ ytr                 # 伪逆 = 最小范数解 (Ch8 SVD)
        risk = float(np.mean((phi(Xte, p) @ beta - yte) ** 2))
        curve.append((p, risk))
        betas[p] = float(np.linalg.norm(beta))

    peak_p, peak_r = max(curve, key=lambda t: t[1])
    sweet = min(r for p, r in curve if p < n)            # 欠参数区甜蜜点
    under = float(np.mean([r for p, r in curve if p < n]))   # 欠参数区均值(含临阈点)
    over = float(np.mean([r for p, r in curve if p >= 300]))  # 深过参数区均值

    # 玩具规模律：良设线性回归 excess risk ~ n^{-1}
    rng2 = np.random.default_rng(1)
    d2, noise2 = 20, 0.3
    w2 = rng2.standard_normal(d2) / np.sqrt(d2)

    def samp2(m):
        X = rng2.standard_normal((m, d2))
        return X, X @ w2 + noise2 * rng2.standard_normal(m)

    XteB, yteB = samp2(20000)
    ns, er = [50, 100, 200, 400, 800, 1600, 3200], []
    for m in ns:
        acc = []
        for _ in range(20):                              # 多次平均，压低测量噪声
            Xtr2, ytr2 = samp2(m)
            b = np.linalg.lstsq(Xtr2, ytr2, rcond=None)[0]
            acc.append(np.mean((XteB @ b - yteB) ** 2) - noise2 ** 2)
        er.append(float(np.mean(acc)))
    slope = float(np.polyfit(np.log(ns), np.log(er), 1)[0])

    return {
        "curve": curve, "betas": betas,
        "peak_p": peak_p, "peak_r": peak_r,
        "sweet": sweet, "under": under, "over": over,
        "n": n, "ns": ns, "er": er, "slope": slope,
    }


def main() -> None:
    r = run()
    for p, risk in r["curve"]:
        print("p=%5d  |beta|=%7.3f  test_risk=%8.4f" % (p, r["betas"][p], risk))

    print("\n            log10(test_risk) →   (*=数据点)")
    lo = np.log10(min(x for _, x in r["curve"]))
    hi = np.log10(max(x for _, x in r["curve"]))
    for p, risk in r["curve"]:
        col = int(round((np.log10(risk) - lo) / (hi - lo) * 39))
        print("p=%5d |" % p + " " * col + "*")

    print("-" * 52)
    print("欠参数区甜蜜点  (p≈20)      风险 ≈ %.2f" % r["sweet"])
    print("欠参数区均值    (含临阈点)  ≈ %.2f" % r["under"])
    print("插值阈值尖峰    p=%d  风险 ≈ %.2f" % (r["peak_p"], r["peak_r"]))
    print("过参数区 (p>>n) 平均测试风险 ≈ %.2f" % r["over"])
    assert r["peak_p"] == r["n"]                          # 尖峰恰在 p = n
    assert r["over"] < r["sweet"]                         # 第二次下降：低于经典甜蜜点
    assert r["over"] < r["peak_r"]                        # 且远低于尖峰

    print("=" * 52)
    for m, e in zip(r["ns"], r["er"]):
        print("n=%5d  excess_risk=%.5f" % (m, e))
    print("log-log 拟合斜率 ≈ %.2f  (理论 -1)" % r["slope"])
    assert r["slope"] < 0


if __name__ == "__main__":
    main()
