"""第 29 章配套代码的回归测试。

注意力 = 可微分的按内容软查找。核心事实（seed=0）：
  1. 注意力权重每行非负且和为 1（合法概率分布）；
  2. lookup 任务：带噪 query 让权重集中到正确槽（准确率高、干净查询正确槽
     ≈0.916），取回正确 value（L2 << V 典型幅度）；
  3. 缩放的必要：不除 √d_k 时 softmax 随维度单调变尖（d_k 越大最大权重越高、
     熵越低，朝 one-hot 塌陷）；除 √d_k 则几乎对维度免疫。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ch29.attention import run, attention, softmax

_R = run()


def test_weights_form_probability_distribution():
    assert abs(_R["row_sum"] - 1.0) < 1e-9, \
        f"注意力权重每行应和为 1，实际 {_R['row_sum']}"
    # 直接验证一个随机例子非负且和为 1
    rng = np.random.default_rng(3)
    Q = rng.standard_normal((4, 8)); K = rng.standard_normal((5, 8)); V = rng.standard_normal((5, 2))
    _, A = attention(Q, K, V)
    assert np.all(A >= 0), "权重应非负"
    assert np.allclose(A.sum(axis=1), 1.0), "每行权重应和为 1"


def test_lookup_retrieves_correct_slot():
    assert _R["accuracy"] > 0.95, f"lookup 准确率应高，实际 {_R['accuracy']}"
    assert _R["avg_max_weight"] > 0.7, \
        f"权重应高度集中，平均最大权重实际 {_R['avg_max_weight']}"
    # 干净查询：正确槽（索引 2）拿走绝大部分注意力
    assert _R["clean_weights"][2] > 0.8, \
        f"干净查询应把权重集中到正确槽，实际 {_R['clean_weights']}"
    # 取回值接近真值，误差远小于 V 的典型幅度（sqrt(var)≈0.94）
    assert _R["retrieved_l2"] < 0.5 * _R["var_V"] ** 0.5 + 0.3, \
        f"取回值应接近真 value，L2={_R['retrieved_l2']}"


def test_scaling_prevents_softmax_saturation():
    s = _R["scaling"]
    # 不缩放：随维度单调变尖（最大权重升、熵降）
    assert s[8]["unscaled_maxw"] < s[64]["unscaled_maxw"] < s[512]["unscaled_maxw"], \
        "不缩放时最大权重应随维度单调上升（softmax 越来越尖）"
    assert s[8]["unscaled_H"] > s[64]["unscaled_H"] > s[512]["unscaled_H"], \
        "不缩放时熵应随维度单调下降（朝 one-hot 塌陷）"
    # 缩放：几乎对维度免疫（最大权重稳定在低位）
    assert s[512]["scaled_maxw"] < 0.3, \
        f"缩放后即便 d_k=512 权重仍弥散，实际 maxw={s[512]['scaled_maxw']}"
    # 高维下缩放 vs 不缩放差距巨大
    assert s[512]["unscaled_maxw"] > 3 * s[512]["scaled_maxw"], \
        "高维下不缩放应远比缩放尖锐"


def test_softmax_numerically_stable():
    # 大输入不溢出、仍是合法分布
    big = np.array([[1000.0, 1001.0, 1002.0]])
    p = softmax(big)
    assert np.all(np.isfinite(p)) and abs(p.sum() - 1.0) < 1e-9, "softmax 应数值稳定"


def _run_all():
    passed = 0
    tests = [
        test_weights_form_probability_distribution,
        test_lookup_retrieves_correct_slot,
        test_scaling_prevents_softmax_saturation,
        test_softmax_numerically_stable,
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
