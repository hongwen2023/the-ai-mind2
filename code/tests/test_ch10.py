"""第 10 章配套代码的回归测试。

  - 蒙特卡洛：大 n 的估计接近真值；估计误差随 n 增大总体减小；π 估计收敛；
  - Hoeffding：经验尾频率处处 ≤ 上界 2exp(-2 n t²)（集中不等式的经验证据）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch10.probability import estimate_coin, estimate_pi, hoeffding_check


def test_monte_carlo_coin_converges():
    rows = estimate_coin()
    n_last, est_last, err_last, _ = rows[-1]
    assert n_last == 1_000_000
    assert err_last < 1e-3, f"n=1e6 的偏币估计应接近 0.37，误差 {err_last}"


def test_error_shrinks_with_n():
    rows = estimate_coin()
    # 首（n=10）误差应远大于末（n=1e6）误差
    assert rows[0][2] > rows[-1][2] * 10, "样本量增大 5 个数量级，误差应显著缩小"


def test_pi_estimate_converges():
    rows = estimate_pi()
    n_last, pi_last, err_last = rows[-1]
    assert err_last < 1e-2, f"n=1e6 的投点估 π 应接近真值，误差 {err_last}"


def test_hoeffding_bound_holds_empirically():
    for n, emp, bound in hoeffding_check():
        assert emp <= bound, f"n={n}: 经验尾频率 {emp} 违反 Hoeffding 上界 {bound}"


def test_hoeffding_bound_shrinks_exponentially():
    rows = hoeffding_check()
    bounds = [b for _, _, b in rows]
    assert all(bounds[i] > bounds[i + 1] for i in range(len(bounds) - 1)), \
        "Hoeffding 上界应随 n 增大而指数下降"


def _run_all():
    passed = 0
    tests = [
        test_monte_carlo_coin_converges,
        test_error_shrinks_with_n,
        test_pi_estimate_converges,
        test_hoeffding_bound_holds_empirically,
        test_hoeffding_bound_shrinks_exponentially,
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
