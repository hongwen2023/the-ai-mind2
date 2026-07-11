"""第 20 章配套代码的回归测试。

GMM-EM 的核心事实（确定性，seed=20）：
  - EM 对数似然单调不减并收敛（ELBO 坐标上升）；
  - 责任每行和为 1（软归属，边界点非 one-hot）；
  - 恢复出接近真值的成分均值。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch20.gmm_em import run

_R = run()


def test_loglik_monotone_nondecreasing():
    assert np.all(np.diff(_R["lls"]) >= -1e-6), "EM 对数似然应单调不减"


def test_loglik_converges():
    lls = _R["lls"]
    assert lls[-1] > lls[0], "最终对数似然应高于初始"
    assert abs(lls[-1] - lls[-2]) < 1.0, "末端对数似然应趋于收敛"


def test_responsibilities_sum_to_one():
    assert _R["resp_rowsum_err"] < 1e-9, "每个点的责任应和为 1"


def test_soft_assignment_on_boundary():
    # 责任熵最大的边界点，其最大责任应明显小于 1（软分配，非 one-hot）
    assert _R["max_resp_of_top_boundary"] < 0.9, \
        f"边界点应是软分配（最大责任<0.9），实际 {_R['max_resp_of_top_boundary']:.3f}"


def test_recovers_true_means():
    # 恢复的均值集合应与真均值集合匹配（顺序可能不同）
    mu, mu_true = _R["mu"], _R["mu_true"]
    for mt in mu_true:
        dists = np.linalg.norm(mu - mt, axis=1)
        assert dists.min() < 0.6, f"应恢复出接近真均值 {mt} 的成分，最近距离 {dists.min():.3f}"


def _run_all():
    passed = 0
    tests = [
        test_loglik_monotone_nondecreasing,
        test_loglik_converges,
        test_responsibilities_sum_to_one,
        test_soft_assignment_on_boundary,
        test_recovers_true_means,
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
