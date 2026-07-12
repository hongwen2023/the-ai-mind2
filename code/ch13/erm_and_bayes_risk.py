"""The AI Mind · 第 13 章配套代码：学习问题的形式化——损失、风险与 ERM

在一个**我们钦定的（因而 R* 可精确算）**分类分布上，撞见贝叶斯地板与风险分解：
  - 已知 η(x)=P(y=1|x)=σ(f(x))，贝叶斯风险 R*=E_x[min(η,1-η)]（海量蒙特卡洛）；
  - 用容量递增的多项式逻辑回归跑 ERM（交叉熵代理），在海量测试集估真实 0-1 风险；
  - 验证**所有模型的真实风险都 ≥ R***（无人越得过贝叶斯地板），
    经验风险 < 真实风险（乐观偏差），提升容量在欠拟合区改善真实风险。

纯 NumPy、固定种子。
运行 `python -m ch13.erm_and_bayes_risk` 打印正文那张表。
"""
from __future__ import annotations

import numpy as np

DEGREES = [1, 2, 4, 6]


def f(x):
    return 1.5 * (x ** 2 - 2.0)   # 决策边界在 x²=2


def eta(x):
    return 1.0 / (1.0 + np.exp(-f(x)))


def sample(n, gen):
    x = gen.uniform(-3.0, 3.0, size=n)
    y = (gen.uniform(size=n) < eta(x)).astype(float)
    return x, y


def bayes_risk(n=2_000_000, seed=0):
    gen = np.random.default_rng(seed)
    x = gen.uniform(-3.0, 3.0, size=n)
    e = eta(x)
    return float(np.mean(np.minimum(e, 1.0 - e)))


def poly(x, d):
    return np.stack([x ** k for k in range(1, d + 1)], axis=1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, iters=4000, lr=0.5):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        w -= lr * (Xb.T @ (sigmoid(Xb @ w) - y) / len(y))
    return w


def zero_one_risk(w, X, y):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return float(np.mean((Xb @ w > 0).astype(float) != y))


def run():
    """返回 (R*, [(d, emp_risk, true_risk)])。"""
    R_star = bayes_risk()
    x_tr, y_tr = sample(200, np.random.default_rng(1))
    x_te, y_te = sample(200_000, np.random.default_rng(2))
    rows = []
    for d in DEGREES:
        Xtr, Xte = poly(x_tr, d), poly(x_te, d)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        w = fit_logistic(Xtr, y_tr)
        rows.append((d, zero_one_risk(w, Xtr, y_tr), zero_one_risk(w, Xte, y_te)))
    return R_star, rows


def main() -> None:
    R_star, rows = run()
    print(f"Bayes risk  R* = {R_star:.4f}")
    print(f"{'d':>2} {'emp R^':>9} {'true R':>9} {'>= R*?':>7}")
    for d, emp, true in rows:
        print(f"{d:>2} {emp:>9.4f} {true:>9.4f} {str(true >= R_star - 0.01):>7}")


if __name__ == "__main__":
    main()
