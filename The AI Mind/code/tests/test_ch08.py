"""第 8 章配套代码的回归测试。

线性代数的几何事实，确定性、可精确断言：
  - [[2,1],[1,2]] 的奇异值是 [3,1]；单位圆的像最大长度=3、最小=1（=最大/最小拉伸）；
    对称矩阵 U=V；
  - Eckart–Young：秩-k 逼近的 Frobenius 误差 == sqrt(丢弃奇异值平方和)；
  - 奇异值非增；完整 SVD 重构 allclose。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch08.geometry_of_linear_algebra import (
    ellipse_demo, make_low_rank_matrix, eckart_young_table, low_rank,
)


def test_singular_values_are_axis_lengths():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    S, mx, mn, U, Vt = ellipse_demo(A)
    assert np.allclose(S, [3.0, 1.0]), f"奇异值应为 [3,1]，实际 {S}"
    assert abs(mx - 3.0) < 1e-9, "椭圆最大半轴 = 最大奇异值 3"
    assert abs(mn - 1.0) < 1e-9, "椭圆最小半轴 = 最小奇异值 1"


def test_symmetric_matrix_U_equals_V():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    _, _, _, U, Vt = ellipse_demo(A)
    assert np.allclose(U, Vt.T), "对称矩阵的 SVD 应满足 U=V（谱定理是 SVD 特例）"


def test_singular_values_non_increasing():
    M = make_low_rank_matrix()
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    assert np.all(np.diff(S) <= 1e-12), "奇异值应从大到小排列"


def test_eckart_young_error_formula():
    M = make_low_rank_matrix()
    _, rows = eckart_young_table(M)
    for k, err, ey in rows:
        assert abs(err - ey) < 1e-9, \
            f"秩-{k} 逼近 Frobenius 误差 {err} 应等于 sqrt(尾部σ平方和) {ey}"


def test_low_rank_error_decreases_with_k():
    M = make_low_rank_matrix()
    _, rows = eckart_young_table(M)
    errs = [err for _, err, _ in rows]
    assert all(errs[i] >= errs[i + 1] - 1e-12 for i in range(len(errs) - 1)), \
        "秩-k 逼近误差应随 k 增大单调不增"


def test_full_svd_reconstructs():
    M = make_low_rank_matrix()
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    assert np.allclose(low_rank(U, S, Vt, len(S)), M), "完整 SVD 应重构原矩阵"


def _run_all():
    passed = 0
    tests = [
        test_singular_values_are_axis_lengths,
        test_symmetric_matrix_U_equals_V,
        test_singular_values_non_increasing,
        test_eckart_young_error_formula,
        test_low_rank_error_decreases_with_k,
        test_full_svd_reconstructs,
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
