"""第 22 章配套代码的回归测试。

从感知机到多层网络的三条核心事实（确定性，seed=0）：
  1. 线性模型（逻辑回归）在 XOR 上准确率 ≤ 0.75（线性不可分，表达力上界）；
  2. 含非线性的 2 层 MLP 解 XOR，准确率 = 1.0；
  3. 隐藏层把 XOR 变成线性可分（隐藏表示上一个线性分类器即可 100% 分开）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch22.perceptron_to_mlp import train_linear_xor, mlp_xor, hidden_is_linearly_separable


def test_linear_model_fails_xor():
    _, _, _, acc = train_linear_xor()
    assert acc <= 0.75, f"线性模型在 XOR 上准确率应 ≤ 0.75（线性不可分），实际 {acc}"


def test_mlp_solves_xor():
    _, pred, acc = mlp_xor()
    assert acc == 1.0, f"含非线性的 MLP 应完美解 XOR，实际 {acc}"
    assert list(pred) == [0.0, 1.0, 1.0, 0.0], "MLP 输出应正是 XOR"


def test_hidden_representation_is_linearly_separable():
    H, _, _ = mlp_xor()
    acc = hidden_is_linearly_separable(H)
    assert acc == 1.0, \
        f"隐藏层应把 XOR 变成线性可分（线性分类器 100%），实际 {acc}"


def _run_all():
    passed = 0
    tests = [
        test_linear_model_fails_xor,
        test_mlp_solves_xor,
        test_hidden_representation_is_linearly_separable,
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
