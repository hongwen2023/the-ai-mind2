"""第 17 章配套代码的回归测试。

核方法与 SVM 的核心事实（确定性，seed=0）：
  - 核技巧：RBF 核在非线性可分数据上准确率远高于线性核；
  - 支持向量稀疏（KKT）：RBF 的支持向量数 ≪ n；
  - 核矩阵合法性（Mercer）：RBF 核矩阵对称且半正定（最小特征值≈0）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch17.kernels_and_svm import run

_R = run()


def test_rbf_beats_linear_on_nonlinear_data():
    assert _R["acc_rbf"] > 0.95, f"RBF 核应在同心圆上高准确率，实际 {_R['acc_rbf']:.3f}"
    assert _R["acc_rbf"] > _R["acc_linear"] + 0.3, \
        f"RBF 应远胜线性核：rbf={_R['acc_rbf']:.3f} linear={_R['acc_linear']:.3f}"


def test_support_vectors_are_sparse():
    assert _R["sv_rbf"] < _R["n"] * 0.5, \
        f"RBF 支持向量应稀疏（≪n），实际 {_R['sv_rbf']}/{_R['n']}"
    # 自由 SV + 有界 SV 应等于总 SV（KKT 分型自洽）
    assert _R["free_sv"] + _R["bounded_sv"] == _R["sv_rbf"]


def test_kernel_matrix_symmetric():
    assert _R["K_sym_err"] < 1e-9, "RBF 核矩阵应对称"


def test_kernel_matrix_psd():
    assert _R["K_min_eig"] > -1e-8, \
        f"RBF 核矩阵应半正定（Mercer），最小特征值 {_R['K_min_eig']:.2e}"


def _run_all():
    passed = 0
    tests = [
        test_rbf_beats_linear_on_nonlinear_data,
        test_support_vectors_are_sparse,
        test_kernel_matrix_symmetric,
        test_kernel_matrix_psd,
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
