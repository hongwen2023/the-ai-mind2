"""第 1 章配套代码的回归测试。

设计原则（直接回应自我评估里"招牌数值 8287 完全可复现是过度承诺"这一条）：
  - 低次数（良态区，d<=5）：用容差断言 R_hat / R_true 的绝对值。
  - 高次数（近病态区，d>=6）：**只断言趋势**（先降后爆、过拟合），不锁死绝对值——
    因为高次 Vandermonde 系统的 true risk 量级对 BLAS/LAPACK/NumPy 版本敏感。
这样测试既守住"过拟合戏剧成立"这一教学结论，又不对跨环境不稳定的数字做虚假承诺。

运行：
  pytest              # 若已安装 pytest
  python test_ch01.py # 无 pytest 时的兜底自检（打印 PASS/FAIL）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch01.learning_problem import run_table, empirical_risk, fit_erm, make_dataset

SIGMA = 0.15
NOISE_FLOOR = SIGMA**2  # 0.0225，Bayes 风险下界

# 参考值：在 numpy 2.4.x / seed=0, n=15, sigma=0.15 下实测得到（正文表）。
# 仅用于低次数良态区的容差断言。
REF_LOW = {
    #  d: (R_hat,  R_true)
    0: (0.3242, 0.5833),
    1: (0.1848, 0.2306),
    2: (0.1701, 0.2355),
    3: (0.0150, 0.0397),
    4: (0.0126, 0.0733),
    5: (0.0105, 0.0510),
}


def _table():
    return {d: (rhat, rtrue) for d, rhat, rtrue in run_table(max_degree=9, seed=0, n=15, sigma=SIGMA)}


def test_low_degree_values_match_reference():
    """低次数（d<=5，良态）：R_hat 与 R_true 应与参考值在容差内吻合。"""
    t = _table()
    for d, (rhat_ref, rtrue_ref) in REF_LOW.items():
        rhat, rtrue = t[d]
        assert abs(rhat - rhat_ref) < 2e-3, f"d={d}: R_hat={rhat:.4f} 偏离参考 {rhat_ref}"
        assert abs(rtrue - rtrue_ref) < 5e-3, f"d={d}: R_true={rtrue:.4f} 偏离参考 {rtrue_ref}"


def test_empirical_risk_monotone_nonincreasing():
    """经验风险 R_hat 随 d 单调不增（嵌套假设空间的可证性质）。允许浮点级别的并列。"""
    t = _table()
    rhats = [t[d][0] for d in range(0, 10)]
    for d in range(1, 10):
        assert rhats[d] <= rhats[d - 1] + 1e-6, f"R_hat 在 d={d} 处上升：{rhats[d-1]:.5f} -> {rhats[d]:.5f}"


def test_true_risk_minimized_at_degree_3():
    """真实风险 R 的最小点在 d=3（这个特定训练集上的"刚刚好"复杂度）。"""
    t = _table()
    rtrues = {d: t[d][1] for d in range(0, 10)}
    best_d = min(rtrues, key=rtrues.get)
    assert best_d == 3, f"真实风险最小点应在 d=3，实际在 d={best_d}"


def test_overfitting_blowup_trend_only():
    """过拟合趋势（不锁死绝对值）：高次真实风险应远大于甜点，且随 d 灾难性放大。"""
    t = _table()
    r3 = t[3][1]
    r6, r9 = t[6][1], t[9][1]
    assert r6 > 10 * r3, f"d=6 的真实风险未显著恶化：R(6)={r6:.4f} vs R(3)={r3:.4f}"
    assert r9 > r6, f"过拟合未随 d 继续放大：R(9)={r9:.4f} 应大于 R(6)={r6:.4f}"
    assert r9 > 100, f"d=9 未出现灾难性过拟合：R(9)={r9:.4f}"


def test_high_degree_fits_below_noise_floor():
    """d=9 的训练误差应低于噪声地板 sigma^2=0.0225（开始把噪声当信号背下来），
    但尚未真正插值（n=15 需 d=14 才能 R_hat=0）。"""
    x, y = make_dataset(seed=0, n=15, sigma=SIGMA)
    rhat9 = empirical_risk(fit_erm(x, y, 9), x, y, 9)
    assert rhat9 < NOISE_FLOOR, f"R_hat(d=9)={rhat9:.4f} 应低于噪声地板 {NOISE_FLOOR}"
    assert rhat9 > 0.0, "d=9 不应恰好插值（那需要 d=14）"


def _run_all():
    passed = 0
    tests = [
        test_low_degree_values_match_reference,
        test_empirical_risk_monotone_nonincreasing,
        test_true_risk_minimized_at_degree_3,
        test_overfitting_blowup_trend_only,
        test_high_degree_fits_below_noise_floor,
    ]
    for tfn in tests:
        try:
            tfn()
            print(f"PASS  {tfn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {tfn.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
