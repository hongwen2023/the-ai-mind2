"""第 11 章配套代码的回归测试。

偏差-方差分解的三条核心事实（确定性、seed=0）：
  1. 分解成立：bias² + var + noise ≈ 期望测试 MSE（良态 d 上 gap 很小）；
  2. 偏差²随复杂度 d 总体下降（高偏差→低偏差）；
  3. 方差随复杂度 d 总体上升（低方差→高方差，d=9 剧增）；
  4. 总误差 U 型：最低点在中段（既非最简也非最繁）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch11.bias_variance import decompose, DEGREES


def _rows():
    return decompose()


def test_decomposition_holds():
    for d, b, v, nz, tot, mse, gap in _rows():
        # 良态复杂度上分解应很紧；d=9 方差重尾，容差放宽
        tol = 0.05 if d <= 7 else 0.1
        assert gap < tol, f"d={d}: bias²+var+noise={tot:.4f} 与 MSE={mse:.4f} 差 {gap:.4f} 过大"


def test_bias_decreases_with_complexity():
    rows = _rows()
    b0 = rows[0][1]        # d=0 偏差²
    b_mid = rows[3][1]     # d=3 偏差²
    assert b0 > b_mid * 10, f"偏差²应随复杂度大幅下降：d=0 {b0:.4f} vs d=3 {b_mid:.4f}"


def test_variance_increases_with_complexity():
    rows = _rows()
    v_simple = rows[0][2]   # d=0 方差
    v_complex = rows[-1][2] # d=9 方差
    assert v_complex > v_simple * 10, f"方差应随复杂度大幅上升：d=0 {v_simple:.4f} vs d=9 {v_complex:.4f}"


def test_total_error_is_u_shaped():
    rows = _rows()
    totals = [tot for _, _, _, _, tot, _, _ in rows]
    argmin = totals.index(min(totals))
    assert 0 < argmin < len(rows) - 1, f"总误差最低点应在中段（U 型），实际在索引 {argmin}"


def _run_all():
    passed = 0
    tests = [
        test_decomposition_holds,
        test_bias_decreases_with_complexity,
        test_variance_increases_with_complexity,
        test_total_error_is_u_shaped,
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
