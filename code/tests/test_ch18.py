"""第 18 章配套代码的回归测试。

树与集成的三条核心事实（确定性，seed=42/1/2）：
  1. 单棵深树过拟合：train 误差≈0，但 test 误差明显更高（高方差）；
  2. 随机森林降方差：test 误差 < 单棵深树 test 误差；
  3. 梯度提升降偏差：test_mse 随轮数总体下降（100 轮 ≪ 1 轮）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch18.trees_forests_boosting import run

_R = run()


def test_single_tree_overfits():
    assert _R["tree_train"] < 0.02, f"单棵深树训练误差应≈0，实际 {_R['tree_train']:.3f}"
    assert _R["tree_test"] > _R["tree_train"] + 0.1, \
        f"单棵深树应过拟合（test≫train）：train={_R['tree_train']:.3f} test={_R['tree_test']:.3f}"


def test_random_forest_reduces_variance():
    assert _R["forest_test"] < _R["tree_test"], \
        f"随机森林 test 误差应低于单棵深树：forest={_R['forest_test']:.3f} tree={_R['tree_test']:.3f}"


def test_boosting_reduces_error_over_rounds():
    b = _R["boost_test_mse"]
    assert b[100] < b[1], f"梯度提升 test_mse 应随轮数下降：1 轮={b[1]:.3f} 100 轮={b[100]:.3f}"
    assert b[20] < b[5] < b[1], "test_mse 应随轮数单调下降（前段）"
    assert b[100] < 0.1, "充分提升后 test_mse 应较小"


def _run_all():
    passed = 0
    tests = [
        test_single_tree_overfits,
        test_random_forest_reduces_variance,
        test_boosting_reduces_error_over_rounds,
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
