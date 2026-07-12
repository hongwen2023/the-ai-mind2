"""第 6 章配套代码的回归测试。

三条定性结论是本章教学核心，稳健、必须成立：
  1. k=1 训练误差恒为 0（每个训练点最近邻是自己——记忆，不是学习）；
  2. 存在某个 k>1，其测试误差明显低于 k=1（背得最熟 ≠ 泛化最好）；
  3. 测试误差随 k 先降后升（最优 k 在中段，不在两端）。
外加对整张表的容差断言（seed=6 下实测逐行吻合）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch06.induction_first_lesson import run_table, KS

# seed=6 下实测的参考表 (train_err, test_err)
REF = {
    1: (0.000, 0.096), 3: (0.040, 0.076), 5: (0.045, 0.080),
    9: (0.052, 0.071), 15: (0.058, 0.069), 25: (0.060, 0.062),
    51: (0.056, 0.058), 101: (0.064, 0.068), 201: (0.080, 0.083),
    401: (0.160, 0.175),
}


def _table():
    return {k: (tr, te) for k, tr, te in run_table()}


def test_k1_train_error_is_zero():
    t = _table()
    assert t[1][0] == 0.0, f"k=1 训练误差应恒为 0（纯记忆），实际 {t[1][0]}"


def test_some_k_generalizes_better_than_k1():
    t = _table()
    test_errs = {k: t[k][1] for k in KS}
    best_k = min(test_errs, key=test_errs.get)
    assert best_k > 1, f"最优（测试误差最低）的 k 应 >1，实际 {best_k}"
    assert test_errs[best_k] < test_errs[1], "存在 k>1 的测试误差低于 k=1"


def test_test_error_is_u_shaped():
    t = _table()
    test_errs = [t[k][1] for k in KS]
    argmin = test_errs.index(min(test_errs))
    assert 0 < argmin < len(KS) - 1, "测试误差最低点应在中段（U 型），不在两端"


def test_table_matches_reference_within_tolerance():
    t = _table()
    for k, (tr_ref, te_ref) in REF.items():
        tr, te = t[k]
        assert abs(tr - tr_ref) < 0.02, f"k={k} 训练误差 {tr:.3f} 偏离参考 {tr_ref}"
        assert abs(te - te_ref) < 0.02, f"k={k} 测试误差 {te:.3f} 偏离参考 {te_ref}"


def _run_all():
    passed = 0
    tests = [
        test_k1_train_error_is_zero,
        test_some_k_generalizes_better_than_k1,
        test_test_error_is_u_shaped,
        test_table_matches_reference_within_tolerance,
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
