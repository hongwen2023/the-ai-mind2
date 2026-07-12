"""The AI Mind · 第 11 章配套代码：统计推断——偏差-方差分解

把期望测试 MSE 数值地拆成 bias² + variance + noise，验证 Ch1/Ch6 的 U 型误差曲线
其实是两个此消彼长分量之和：随多项式次数 d 增大，偏差²下降、方差上升，总误差在中段最小。

做法：已知真函数 f + 噪声 sigma；反复抽 n_sets 个训练集，对每个复杂度 d 拟合，
在固定测试点收集预测，据训练集间的分布估计三个分量。

纯 NumPy、固定种子。
运行 `python -m ch11.bias_variance` 打印正文那张分解表。
"""
from __future__ import annotations

import numpy as np

DEGREES = [0, 1, 2, 3, 5, 7, 9]


def f_true(x):
    return np.sin(2.0 * np.pi * x)


def design(x, d):
    xs = 2.0 * x - 1.0  # 映到 [-1,1] 改善条件数（回指 Ch9）
    return np.vander(xs, d + 1, increasing=True)


def fit_predict(x_tr, y_tr, x_te, d):
    coef, *_ = np.linalg.lstsq(design(x_tr, d), y_tr, rcond=None)
    return design(x_te, d) @ coef


def decompose(seed=0, sigma=0.30, n_train=30, n_sets=500):
    """返回 [(d, bias2, var, noise, sum, mse, gap)]。"""
    rng = np.random.default_rng(seed)
    x_test = np.linspace(0.05, 0.95, 40)
    f_test = f_true(x_test)
    noise = sigma ** 2
    rows = []
    for d in DEGREES:
        preds = np.empty((n_sets, x_test.size))
        mse_acc = np.zeros(x_test.size)
        for s in range(n_sets):
            x_tr = rng.random(n_train)
            y_tr = f_true(x_tr) + sigma * rng.standard_normal(n_train)
            p = fit_predict(x_tr, y_tr, x_test, d)
            preds[s] = p
            y_te = f_test + sigma * rng.standard_normal(x_test.size)
            mse_acc += (p - y_te) ** 2
        mean_pred = preds.mean(axis=0)
        bias2 = float(np.mean((mean_pred - f_test) ** 2))
        var = float(np.mean(preds.var(axis=0)))
        mse = float(np.mean(mse_acc / n_sets))
        tot = bias2 + var + noise
        rows.append((d, bias2, var, noise, tot, mse, abs(tot - mse)))
    return rows


def main() -> None:
    print(f"{'d':>3} {'bias^2':>9} {'var':>9} {'noise':>7} {'sum':>9} {'MSE':>9} {'gap':>7}")
    for d, b, v, nz, tot, mse, gap in decompose():
        print(f"{d:>3} {b:9.4f} {v:9.4f} {nz:7.4f} {tot:9.4f} {mse:9.4f} {gap:7.4f}")


if __name__ == "__main__":
    main()
