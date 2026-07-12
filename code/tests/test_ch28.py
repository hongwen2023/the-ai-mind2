"""第 28 章配套代码的回归测试。

从零复现双下降的核心事实（seed=0，确定性）：
  1. 尖峰恰在插值阈值 p=n=40，且高（≈13.1）——σ_min→0 导致方差爆炸；
  2. 过参数区 (p>=300) 平均风险 ≈0.40，**低于**经典甜蜜点 ≈0.82——双下降强形式；
  3. 最小范数解范数 |beta| 在 p=n 处飙高、在深过参数区跌回很小（倒过来的双下降）；
  4. 玩具规模律：良设线性回归超额风险随 n 下降，log-log 斜率为负（≈-0.95）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch28.science_of_dl import run

_R = run()


def test_peak_at_interpolation_threshold():
    assert _R["peak_p"] == _R["n"], \
        f"双下降尖峰应恰在插值阈值 p=n={_R['n']}，实际 p={_R['peak_p']}"
    assert _R["peak_r"] > 5.0, \
        f"插值阈值尖峰应显著高（σ_min→0 方差爆炸），实际 {_R['peak_r']:.2f}"


def test_second_descent_below_sweet_spot():
    # 双下降的定义性性质：过参数区低于经典甜蜜点
    assert _R["over"] < _R["sweet"], \
        f"过参数区应低于经典甜蜜点：over={_R['over']:.2f} vs sweet={_R['sweet']:.2f}"
    assert _R["over"] < _R["peak_r"] * 0.1, \
        f"过参数区应远低于尖峰：over={_R['over']:.2f} vs peak={_R['peak_r']:.2f}"


def test_min_norm_solution_norm_blows_up_at_threshold():
    # |beta| 在插值阈值处爆炸、在深过参数区回落（隐式正则的最直白证据）
    b = _R["betas"]
    assert b[40] > 3.0, f"最小范数解范数应在 p=n 飙高，实际 {b[40]:.2f}"
    assert b[2000] < 0.5, f"深过参数区解范数应很小，实际 {b[2000]:.2f}"
    assert b[40] > 10 * b[2000], "范数曲线应呈倒双下降（阈值高、两端低）"


def test_toy_scaling_law_is_power_law():
    # 超额风险随 n 单调下降；log-log 斜率为负（幂律 ~ n^{-1}）
    er = _R["er"]
    assert all(er[i] > er[i + 1] for i in range(len(er) - 1)), \
        "超额风险应随 n 单调下降"
    assert -1.3 < _R["slope"] < -0.7, \
        f"log-log 斜率应≈-1（玩具估计率），实际 {_R['slope']:.2f}"


def _run_all():
    passed = 0
    tests = [
        test_peak_at_interpolation_threshold,
        test_second_descent_below_sweet_spot,
        test_min_norm_solution_norm_blows_up_at_threshold,
        test_toy_scaling_law_is_power_law,
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
