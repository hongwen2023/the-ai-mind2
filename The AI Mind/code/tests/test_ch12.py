"""第 12 章配套代码的回归测试。

信息论恒等式与命名定理，确定性、可精确断言：
  - H(p,q) = H(p) + D(p‖q)；D(p‖q) ≥ 0；D(p‖p) = 0；
  - KL 不对称：D(p‖q) ≠ D(q‖p)；
  - 命名定理：交叉熵 H(p,q) 在 q=p 处最小，且最小值 = H(p)；
  - dyadic 分布下平均最优码长 = 熵。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch12.information_theory import entropy, cross_entropy, kl, scan_cross_entropy

P = np.array([0.5, 0.25, 0.125, 0.125])
Q = np.array([0.25, 0.25, 0.25, 0.25])


def test_cross_entropy_identity():
    assert abs(cross_entropy(P, Q) - (entropy(P) + kl(P, Q))) < 1e-12, \
        "应有 H(p,q) = H(p) + D(p‖q)"


def test_kl_nonnegative_and_zero_iff_equal():
    assert kl(P, Q) >= 0
    assert abs(kl(P, P)) < 1e-12, "D(p‖p) 应为 0"
    q2 = np.array([0.1, 0.2, 0.3, 0.4])
    assert kl(P, q2) > 0, "p≠q 时 KL 应严格为正"


def test_kl_is_asymmetric():
    q2 = np.array([0.1, 0.2, 0.3, 0.4])
    assert abs(kl(P, q2) - kl(q2, P)) > 1e-3, "KL 应不对称"


def test_named_theorem_cross_entropy_minimized_at_q_equals_p():
    rows = scan_cross_entropy(P)
    ces = [ce for _, ce, _ in rows]
    argmin = int(np.argmin(ces))
    assert rows[argmin][0] == 0.0, "交叉熵应在 t=0（即 q=p）处最小"
    assert abs(min(ces) - entropy(P)) < 1e-12, "交叉熵最小值应等于 H(p)"
    assert abs(rows[argmin][2]) < 1e-12, "q=p 时 KL 应为 0"


def test_dyadic_code_length_equals_entropy():
    code_len = -np.log2(P)
    avg = float(np.sum(P * code_len))
    assert abs(avg - entropy(P)) < 1e-12, "dyadic 分布平均最优码长应等于熵"


def _run_all():
    passed = 0
    tests = [
        test_cross_entropy_identity,
        test_kl_nonnegative_and_zero_iff_equal,
        test_kl_is_asymmetric,
        test_named_theorem_cross_entropy_minimized_at_q_equals_p,
        test_dyadic_code_length_equals_entropy,
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
