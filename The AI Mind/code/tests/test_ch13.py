"""第 13 章配套代码的回归测试。

学习问题形式化的三条核心事实（确定性）：
  1. 贝叶斯地板：所有 ERM 拟合模型的真实风险都 ≥ R*（-0.01 吸收蒙特卡洛估计误差）；
  2. 乐观偏差：拟合到位后经验风险 ≤ 真实风险；
  3. 近似误差随容量下降：d=1（线性欠拟合）真实风险远高于 d=2（能表达非线性边界）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch13.erm_and_bayes_risk import run


def test_no_model_beats_bayes_floor():
    R_star, rows = run()
    for d, emp, true in rows:
        assert true >= R_star - 0.01, \
            f"d={d} 的真实风险 {true:.4f} 不应低于贝叶斯地板 R*={R_star:.4f}"


def test_bayes_risk_reasonable():
    R_star, _ = run()
    assert 0.05 < R_star < 0.2, f"该分布的贝叶斯风险应在 (0.05,0.2)，实际 {R_star:.4f}"


def test_capacity_reduces_approximation_error():
    _, rows = run()
    true_by_d = {d: true for d, _, true in rows}
    assert true_by_d[1] > true_by_d[2] + 0.1, \
        f"线性(d=1)欠拟合，真实风险应远高于 d=2：{true_by_d[1]:.4f} vs {true_by_d[2]:.4f}"


def test_empirical_risk_is_optimistic():
    _, rows = run()
    # d>=2 已能拟合，经验风险不应显著高于真实风险（乐观方向）
    for d, emp, true in rows:
        if d >= 2:
            assert emp <= true + 0.02, f"d={d}: 经验风险应偏乐观（≤真实风险），emp={emp} true={true}"


def _run_all():
    passed = 0
    tests = [
        test_no_model_beats_bayes_floor,
        test_bayes_risk_reasonable,
        test_capacity_reduces_approximation_error,
        test_empirical_risk_is_optimistic,
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
