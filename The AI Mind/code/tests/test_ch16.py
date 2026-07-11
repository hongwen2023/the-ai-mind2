"""第 16 章配套代码的回归测试。

线性模型正则化的三条核心性质（确定性，seed=0）：
  (i)  岭回归：||w||₂ 随 λ 单调下降，且从不置零（nnz 恒为全维）；
  (ii) lasso：出现精确的 0，非零个数随 λ 增大而减少（稀疏）；
  (iii) 存在 λ>0 使测试误差低于 λ=0（正则改善泛化）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch16.linear_models import ridge_path, lasso_path


def test_ridge_shrinks_monotonically_never_zeros():
    rows = ridge_path()
    norms = [norm for _, norm, _, _ in rows]
    assert all(norms[i] >= norms[i + 1] for i in range(len(norms) - 1)), \
        "岭回归 ||w||₂ 应随 λ 单调不增"
    assert all(k == 20 for _, _, k, _ in rows), "岭回归只收缩不置零，非零个数应恒为全维 20"


def test_lasso_produces_sparsity():
    rows = lasso_path()
    nnzs = [k for _, _, k, _ in rows]
    assert nnzs[0] == 20, "λ=0 时应稠密"
    assert min(nnzs) <= 3, f"大 λ 时 lasso 应压出稀疏解（≤3 非零），实际最小 {min(nnzs)}"
    assert all(nnzs[i] >= nnzs[i + 1] for i in range(len(nnzs) - 1)), \
        "非零个数应随 λ 增大而不增"


def test_regularization_improves_generalization():
    # 岭：某 λ>0 的测试误差低于 λ=0
    r = ridge_path()
    test0 = r[0][3]
    assert min(row[3] for row in r) < test0, "应存在 λ>0 使岭回归测试误差低于 λ=0"
    # lasso：某 λ>0 的测试误差低于 λ=0
    l = lasso_path()
    assert min(row[3] for row in l) < l[0][3], "应存在 λ>0 使 lasso 测试误差低于 λ=0"


def _run_all():
    passed = 0
    tests = [
        test_ridge_shrinks_monotonically_never_zeros,
        test_lasso_produces_sparsity,
        test_regularization_improves_generalization,
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
