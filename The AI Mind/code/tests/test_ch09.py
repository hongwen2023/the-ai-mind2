"""第 9 章配套代码的回归测试。

  - 梯度检验：本章手推的解析梯度与有限差分吻合（相对误差 < 1e-6）——数值验证 Ch7 欠条；
  - softmax 稳定性：天真版对大 logit 溢出成 nan，log-sum-exp 稳定版给出合法概率（和=1）；
    两者在小输入上一致；
  - 条件数：范德蒙德矩阵随阶数病态爆炸（m=19 时 κ 远超 1e12，回收 Ch5 伏笔）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch09.matrix_calculus_and_numerics import (
    gradient_check, softmax_naive, softmax_stable, vandermonde_cond,
)


def test_gradient_check_passes():
    rel = gradient_check()
    assert rel < 1e-6, f"手推解析梯度应与数值梯度吻合（<1e-6），实际 {rel:.2e}"


def test_naive_softmax_overflows_on_large_logits():
    big = np.array([1000., 1001., 1002.])
    with np.errstate(over="ignore", invalid="ignore"):
        out = softmax_naive(big)
    assert np.isnan(out).any(), "天真 softmax 在大 logit 上应溢出成 nan"


def test_stable_softmax_is_valid_on_large_logits():
    big = np.array([1000., 1001., 1002.])
    out = softmax_stable(big)
    assert not np.isnan(out).any(), "稳定 softmax 不应产生 nan"
    assert abs(out.sum() - 1.0) < 1e-9, "softmax 输出应为概率（和=1）"
    assert np.all(out >= 0), "概率应非负"


def test_naive_and_stable_agree_on_small_inputs():
    small = np.array([1., 2., 3.])
    assert np.allclose(softmax_naive(small), softmax_stable(small)), \
        "小输入上天真版与稳定版应一致（稳定版只是数学等价的重写）"


def test_vandermonde_ill_conditioning_explodes():
    conds = [vandermonde_cond(m) for m in (5, 10, 15, 19)]
    assert all(conds[i] < conds[i + 1] for i in range(len(conds) - 1)), \
        "条件数应随阶数单调增大"
    assert conds[-1] > 1e12, f"m=19 的范德蒙德条件数应极大（回收 Ch5 伏笔），实际 {conds[-1]:.2e}"


def _run_all():
    passed = 0
    tests = [
        test_gradient_check_passes,
        test_naive_softmax_overflows_on_large_logits,
        test_stable_softmax_is_valid_on_large_logits,
        test_naive_and_stable_agree_on_small_inputs,
        test_vandermonde_ill_conditioning_explodes,
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
