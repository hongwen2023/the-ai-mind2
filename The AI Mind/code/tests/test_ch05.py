"""第 5 章配套代码的回归测试。

确定性、无随机，可精确断言。核心是把"表示会撒谎"钉成事实：
  - 整数编码：北京(0)↔广州(2) 距离 = 2，北京↔上海(1) = 1（虚假的两倍远结构）
  - one-hot：任意两个不同类别欧氏距离恒 = sqrt(2)（等距、诚实）
  - one-hot 每行和 = 1；标准化后列均值≈0、标准差≈1；X 形状 (5, 11)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch05.turning_world_into_numbers import (
    build_feature_matrix, city_integer_encoding, city_onehot_encoding,
    pairwise, standardize, AREA, CITY,
)


def test_feature_matrix_shape():
    X, vocab = build_feature_matrix()
    # 2 数值 + 3 one-hot 城市 + 6 词袋 = 11
    assert X.shape == (5, 11), f"X 形状应为 (5,11)，实际 {X.shape}"
    assert len(vocab) == 6


def test_onehot_rows_sum_to_one():
    oh = city_onehot_encoding()
    assert np.allclose(oh.sum(axis=1), 1.0), "one-hot 每行和应为 1"


def test_standardized_stats():
    z = standardize(AREA)
    assert abs(z.mean()) < 1e-12
    assert abs(z.std() - 1.0) < 1e-12


def test_integer_encoding_fabricates_false_distance():
    ci = city_integer_encoding().reshape(-1, 1).astype(float)
    d = pairwise(ci)
    # CITY = [北京, 上海, 北京, 广州, 上海] -> 整数 [0,1,0,2,1]
    assert d[0, 3] == 2.0, "整数编码下 北京↔广州 应为 2（虚假）"
    assert d[0, 1] == 1.0, "整数编码下 北京↔上海 应为 1"
    assert d[0, 3] == 2 * d[0, 1], "整数编码凭空造出'广州两倍远'的假结构"


def test_onehot_all_distinct_categories_equidistant():
    d = pairwise(city_onehot_encoding())
    n = len(CITY)
    ci = city_integer_encoding()
    for i in range(n):
        for j in range(n):
            if ci[i] != ci[j]:
                assert abs(d[i, j] - np.sqrt(2)) < 1e-12, "one-hot 下不同类别距离应恒为 sqrt(2)"
            else:
                assert d[i, j] == 0.0


def _run_all():
    passed = 0
    tests = [
        test_feature_matrix_shape,
        test_onehot_rows_sum_to_one,
        test_standardized_stats,
        test_integer_encoding_fabricates_false_distance,
        test_onehot_all_distinct_categories_equidistant,
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
