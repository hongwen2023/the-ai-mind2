"""第 7 章配套代码的回归测试。

核心断言（Book I 收官的垂直切片：一台真会学习的机器）：
  1. 梯度检验通过——解析梯度与有限差分数值梯度极接近（相对误差 < 1e-5），
     证明"误差×输入"的梯度公式写对了；
  2. 学习确实发生——最终损失显著低于初始损失（初始≈ln2≈0.693）；
  3. 泛化成立——训练/测试准确率都远高于随机猜测 0.5。
这些性质稳健；另加对关键数值的容差断言（seed=7 下实测）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch07.vertical_slice import run_experiment


def test_gradient_check_passes():
    r = run_experiment()
    assert r["grad_check_rel"] < 1e-5, \
        f"梯度检验相对误差应 < 1e-5（解析梯度写对），实际 {r['grad_check_rel']:.2e}"


def test_learning_happened():
    r = run_experiment()
    assert r["final_loss"] < r["init_loss"], "最终损失应低于初始损失（学习发生）"
    assert r["final_loss"] < 0.15, f"最终损失应明显下降，实际 {r['final_loss']:.4f}"
    assert abs(r["init_loss"] - 0.6931) < 1e-3, "初始损失应≈ln2（全 0 参数下的交叉熵）"


def test_generalizes_above_chance():
    r = run_experiment()
    assert r["train_acc"] > 0.9, f"训练准确率应远高于 0.5，实际 {r['train_acc']:.3f}"
    assert r["test_acc"] > 0.9, f"测试准确率应远高于 0.5，实际 {r['test_acc']:.3f}"


def test_final_numbers_within_tolerance():
    # seed=7 下实测参考值
    r = run_experiment()
    assert abs(r["final_loss"] - 0.0496) < 0.01
    assert abs(r["train_acc"] - 0.979) < 0.03
    assert abs(r["test_acc"] - 0.992) < 0.03


def _run_all():
    passed = 0
    tests = [
        test_gradient_check_passes,
        test_learning_happened,
        test_generalizes_above_chance,
        test_final_numbers_within_tolerance,
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
