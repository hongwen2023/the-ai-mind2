"""第 25 章配套代码的回归测试。

表示学习的核心事实（确定性，seed=0）：
  1. 三重基线隔离"学习"：原始输入 ≈ 随机投影 ≈ 瞎猜（升维救不了），
     而 MLP 学到的隐藏表示上的线性探针高准确率——只有"学习"让问题线性可分；
  2. 端到端 MLP 解出这个藏进噪声的 XOR；
  3. 嵌入几何：king-man+woman ≈ queen（类比余弦≈1），且 king 与 queen 比与 woman 更近。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch25.representation_learning import run

_R = run()


def test_learned_representation_beats_raw_and_random():
    assert _R["acc_probe"] > 0.9, \
        f"学到的隐藏表示上线性探针应高准确率，实际 {_R['acc_probe']:.3f}"
    assert _R["acc_probe"] > _R["acc_raw"] + 0.3, \
        f"学到的表示应远胜原始输入：{_R['acc_probe']:.3f} vs {_R['acc_raw']:.3f}"
    assert _R["acc_probe"] > _R["acc_rand"] + 0.3, \
        f"学到的表示应远胜随机投影（升维≠学习）：{_R['acc_probe']:.3f} vs {_R['acc_rand']:.3f}"


def test_raw_and_random_are_near_chance():
    # XOR 线性不可分：原始输入与随机投影上的线性探针都≈瞎猜
    assert _R["acc_raw"] < 0.7, "原始输入上线性探针应≈随机（XOR 线性不可分）"
    assert _R["acc_rand"] < 0.7, "随机投影上线性探针应≈随机（只升维、没学）"


def test_mlp_solves_hidden_xor():
    assert _R["acc_mlp"] > 0.95, f"端到端 MLP 应解出藏进噪声的 XOR，实际 {_R['acc_mlp']:.3f}"


def test_embedding_analogy():
    assert _R["cos_analogy_queen"] > 0.95, \
        f"king-man+woman 应≈queen（类比余弦≈1），实际 {_R['cos_analogy_queen']:.3f}"
    assert _R["cos_king_queen"] > _R["cos_king_woman"], \
        "语义上 king 与 queen 应比与 woman 更近"


def _run_all():
    passed = 0
    tests = [
        test_learned_representation_beats_raw_and_random,
        test_raw_and_random_are_near_chance,
        test_mlp_solves_hidden_xor,
        test_embedding_analogy,
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
