"""第 19 章配套代码的回归测试。

  - k-means：簇内平方和 J 序列单调不增并收敛（块坐标下降）；
  - PCA：前 k 主成分解释绝大部分方差；
    降 k 维重构误差 = 丢弃奇异值平方和（Eckart–Young，回指 Ch8）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch19.unsupervised import run_kmeans, run_pca


def test_kmeans_objective_monotone_nonincreasing():
    history = run_kmeans()
    assert all(a >= b - 1e-9 for a, b in zip(history, history[1:])), \
        "k-means 簇内平方和应单调不增"


def test_kmeans_converges():
    history = run_kmeans()
    assert history[-1] < history[0], "最终 J 应显著低于初始 J"
    assert abs(history[-1] - history[-2]) < 1e-6, "收敛后 J 应稳定"


def test_pca_top_components_capture_variance():
    p = run_pca()
    assert p["top_k_ratio"] > 0.95, \
        f"前 2 主成分应解释绝大部分方差（数据本质 2 维），实际 {p['top_k_ratio']:.4f}"


def test_pca_reconstruction_equals_dropped_singular_values():
    p = run_pca()
    assert np.isclose(p["recon_err"], p["dropped_sv2"], rtol=1e-6), \
        f"重构误差 {p['recon_err']:.4f} 应等于丢弃奇异值平方和 {p['dropped_sv2']:.4f}（Eckart–Young）"


def _run_all():
    passed = 0
    tests = [
        test_kmeans_objective_monotone_nonincreasing,
        test_kmeans_converges,
        test_pca_top_components_capture_variance,
        test_pca_reconstruction_equals_dropped_singular_values,
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
