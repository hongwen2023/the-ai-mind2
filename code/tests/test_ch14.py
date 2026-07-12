"""第 14 章配套代码的回归测试。

优化的核心事实（确定性，seed=0）：
  1. 步长三态：η=1/L 收敛到最优（gap 极小）；η>2/L 发散；η 太小收敛但慢；
  2. 光滑常数 L = A 最大特征值 = 30；
  3. 全批 GD 在良态凸问题上收敛到机器精度；
  4. SGD：固定步长停在噪声球；衰减步长把噪声球压下去（末端更小）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch14.optimization import run


def test_smoothness_constant_is_lambda_max():
    r = run()
    assert abs(r["L"] - 30.0) < 1e-9, "L 应等于 A 的最大特征值 30"


def test_step_size_good_converges():
    r = run()
    good = r["three_states"]["good"]
    assert not good["diverged"] and good["final_gap"] < 1e-6, \
        f"η=1/L 应收敛到最优，末端 gap={good['final_gap']:.2e}"


def test_step_size_too_large_diverges():
    r = run()
    assert r["three_states"]["large"]["diverged"], "η>2/L 应发散"


def test_step_size_too_small_is_slower_than_good():
    r = run()
    small = r["three_states"]["small"]["final_gap"]
    good = r["three_states"]["good"]["final_gap"]
    assert small > good, f"太小步长应收敛更慢（末端 gap 更大）：small={small:.2e} good={good:.2e}"


def test_full_batch_gd_reaches_optimum():
    r = run()
    assert abs(r["full_gd_gap"]) < 1e-8, f"良态凸问题上全批 GD 应收敛到机器精度，实际 {r['full_gd_gap']:.2e}"


def test_decaying_step_beats_fixed_step_sgd():
    r = run()
    assert r["sgd_decay_gap"] < r["sgd_fixed_gap"], \
        f"衰减步长应把噪声球压得比固定步长更低：decay={r['sgd_decay_gap']:.2e} fixed={r['sgd_fixed_gap']:.2e}"
    assert r["sgd_fixed_gap"] < 1e-2, "固定步长 SGD 也应大致收敛到最优附近"


def _run_all():
    passed = 0
    tests = [
        test_smoothness_constant_is_lambda_max,
        test_step_size_good_converges,
        test_step_size_too_large_diverges,
        test_step_size_too_small_is_slower_than_good,
        test_full_batch_gd_reaches_optimum,
        test_decaying_step_beats_fixed_step_sgd,
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
