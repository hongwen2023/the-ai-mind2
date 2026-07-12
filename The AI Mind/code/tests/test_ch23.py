"""第 23 章配套代码的回归测试（micro-autograd 里程碑）。

  1. 引擎正确：autograd 梯度与有限差分数值梯度一致（相对误差极小）；
  2. fan-out：一个节点多次使用时梯度正确相加（L=b·b+b, b=-1 → dL/db=-1）；
  3. 里程碑：用 micro-autograd 训练的 MLP 在 XOR 上损失下降、准确率=1.0
     （Ch22 搭好却训不了的网络，现在学会了）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch23.micrograd import Value, grad_check, train_xor


def test_gradient_check_passes():
    rel_a, rel_b = grad_check()
    assert rel_a < 1e-6 and rel_b < 1e-6, \
        f"autograd 梯度应与有限差分一致，rel_a={rel_a:.1e} rel_b={rel_b:.1e}"


def test_fanout_gradients_accumulate():
    # L = b*b + b, b=-1 → dL/db = 2b + 1 = -1（两条路径的贡献相加）
    b = Value(-1.0)
    L = b * b + b
    L.backward()
    assert abs(b.grad - (-1.0)) < 1e-12, \
        f"fan-out 梯度应相加得 -1（若用 = 覆盖会得 +1），实际 {b.grad}"


def test_add_and_mul_local_gradients():
    x, y = Value(3.0), Value(4.0)
    z = x * y + x           # dz/dx = y + 1 = 5, dz/dy = x = 3
    z.backward()
    assert abs(x.grad - 5.0) < 1e-12
    assert abs(y.grad - 3.0) < 1e-12


def test_milestone_mlp_learns_xor():
    losses, acc = train_xor()
    assert losses[-1] < losses[0] * 0.1, \
        f"训练损失应大幅下降：{losses[0]:.3f} → {losses[-1]:.3f}"
    assert acc == 1.0, f"micro-autograd 训练的 MLP 应解出 XOR（准确率 1.0），实际 {acc}"


def _run_all():
    passed = 0
    tests = [
        test_gradient_check_passes,
        test_fanout_gradients_accumulate,
        test_add_and_mul_local_gradients,
        test_milestone_mlp_learns_xor,
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
