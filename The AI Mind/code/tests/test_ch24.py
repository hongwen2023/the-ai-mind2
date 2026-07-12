"""第 24 章配套代码的回归测试。

训练的科学的三条核心事实（确定性）：
  1. 全零初始化因对称性学不动：损失停在 ln2≈0.693、测试准确率≈随机 0.5、
     隐藏层各神经元输出完全相同（列 std = 0）；
  2. He 初始化训练到低损失、高准确率；
  3. Adam 在同样步数下到更低的损失（< 全批 GD）。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch24.training import run

_R = run()


def test_zero_init_fails_by_symmetry():
    assert abs(_R["zero_loss"] - math.log(2)) < 1e-3, \
        f"全零初始化损失应停在 ln2≈0.693，实际 {_R['zero_loss']:.4f}"
    assert _R["zero_acc"] < 0.6, f"全零初始化准确率应≈随机，实际 {_R['zero_acc']:.3f}"
    assert _R["zero_hidden_std"] < 1e-9, \
        "全零初始化下隐藏层各神经元输出应完全相同（对称性未破缺，列 std=0）"


def test_he_init_trains():
    assert _R["he_loss"] < 0.2, f"He 初始化应训练到低损失，实际 {_R['he_loss']:.4f}"
    assert _R["he_acc"] > 0.85, f"He 初始化应达到高准确率，实际 {_R['he_acc']:.3f}"


def test_adam_beats_gd():
    assert _R["adam_final"] < _R["gd_final"], \
        f"同样步数下 Adam 应到更低损失：Adam {_R['adam_final']:.4f} vs GD {_R['gd_final']:.4f}"


def _run_all():
    passed = 0
    tests = [test_zero_init_fails_by_symmetry, test_he_init_trains, test_adam_beats_gd]
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
