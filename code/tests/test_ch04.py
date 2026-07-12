"""第 4 章配套代码的回归测试。

核心断言：循环版与向量化版是"同一个想法的两种表达"——结果必须一致（np.allclose）。
计数为确定性整数，可精确断言；浮点结果用 allclose 而非 ==。

运行：pytest   或   python test_ch04.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch04.thinking_in_code import (
    make_data, standardize_loop, standardize_vec,
    pairwise_dist_loop, pairwise_dist_vec,
)


def test_standardize_loop_equals_vectorized():
    X = make_data()
    Z_loop, _ = standardize_loop(X)
    Z_vec = standardize_vec(X)
    assert np.allclose(Z_loop, Z_vec), "循环版与向量化版标准化结果应一致"


def test_standardize_scalar_count():
    X = make_data(n=6, d=3)
    _, n_scalar = standardize_loop(X)
    assert n_scalar == 6 * 3, f"标准化步标量赋值应为 n*d=18，实际 {n_scalar}"


def test_standardized_stats():
    X = make_data()
    Z = standardize_vec(X)
    assert np.allclose(Z.mean(axis=0), 0.0), "标准化后每列均值应≈0"
    assert np.allclose(Z.std(axis=0), 1.0), "标准化后每列标准差应≈1"


def test_broadcasting_shape():
    a = np.arange(3).reshape(3, 1)
    b = np.arange(4).reshape(1, 4)
    assert (a + b).shape == (3, 4), "(3,1)+(1,4) 应广播成 (3,4)"


def test_pairwise_distance_loop_equals_vectorized():
    X = make_data()
    assert np.allclose(pairwise_dist_vec(X), pairwise_dist_loop(X)), "成对距离两版应一致"
    D = pairwise_dist_vec(X)
    assert np.allclose(np.diag(D), 0.0), "对角线（自距离）应为 0"


def _run_all():
    passed = 0
    tests = [
        test_standardize_loop_equals_vectorized,
        test_standardize_scalar_count,
        test_standardized_stats,
        test_broadcasting_shape,
        test_pairwise_distance_loop_equals_vectorized,
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
