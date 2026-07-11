"""第 15 章配套代码的回归测试（Book II 收官）。

泛化理论的三条可测断言（确定性）：
  1. 一致收敛：最大偏差随 n 减小；max_dev·√n 大致恒定（1/√n 标度）；
     有限类泛化界处处 ≥ 实测最大偏差（Hoeffding+union bound 成立）；
  2. 泛化差距随容量增大而扩大（高容量插值时 R̂≈0、差距巨大）；
  3. 随机标签：高容量模型训练误差≈0（记忆），测试误差≈随机 0.5。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch15.learning_theory import run

_RES = run()


def test_uniform_convergence_shrinks_with_n():
    rows = _RES["uniform"]
    devs = [md for _, md, _, _ in rows]
    assert devs[0] > devs[-1] * 3, f"最大偏差应随 n 显著减小：{devs[0]:.3f} → {devs[-1]:.3f}"


def test_one_over_sqrt_n_scaling():
    rows = _RES["uniform"]
    scaled = [mds for _, _, mds, _ in rows]  # max_dev * sqrt(n)
    # 若确为 1/√n，该乘积应大致恒定（落在一个窄带里）
    assert max(scaled) / min(scaled) < 1.5, f"max_dev·√n 应大致恒定（1/√n 标度），实际 {scaled}"


def test_generalization_bound_holds():
    rows = _RES["uniform"]
    for n, md, _, bound in rows:
        assert bound >= md, f"n={n}: 有限类泛化界 {bound:.4f} 应 ≥ 实测最大偏差 {md:.4f}"


def test_gap_grows_with_capacity():
    rows = _RES["capacity"]
    gap_low = rows[0][3]      # n_feat=2
    gap_high = rows[-1][3]    # n_feat=119（插值）
    assert gap_high > gap_low + 0.2, f"高容量泛化差距应远大于低容量：{gap_low:.3f} vs {gap_high:.3f}"
    assert rows[-1][1] < 0.02, "高容量插值时训练误差应≈0"


def test_random_labels_are_memorized_not_learned():
    train_err, test_err = _RES["random_labels"]
    assert train_err < 0.05, f"高容量模型应能记住随机标签（训练误差≈0），实际 {train_err:.3f}"
    assert test_err > 0.4, f"随机标签下测试误差应≈随机 0.5，实际 {test_err:.3f}"


def _run_all():
    passed = 0
    tests = [
        test_uniform_convergence_shrinks_with_n,
        test_one_over_sqrt_n_scaling,
        test_generalization_bound_holds,
        test_gap_grows_with_capacity,
        test_random_labels_are_memorized_not_learned,
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
