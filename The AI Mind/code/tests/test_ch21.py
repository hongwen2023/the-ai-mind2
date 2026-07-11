"""第 21 章配套代码的回归测试。

评估手艺的核心事实（确定性，seed=42/7）：
  1. AUC 两种独立算法一致（梯形积分 = 排序概率定义）；
  2. precision/recall/F1 计算正确（F1 = 2PR/(P+R)）；
  3. 准确率会骗人：极不平衡下多数类基线准确率≈99% 但召回=0，
     真分类器准确率略低却召回很高；
  4. k 折 CV 各折有方差（单次留出不可靠）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch21.evaluation import run

_R = run()


def test_auc_two_algorithms_agree():
    assert abs(_R["auc_trap"] - _R["auc_prob"]) < 1e-9, \
        f"梯形 AUC {_R['auc_trap']} 应等于排序概率 AUC {_R['auc_prob']}"
    assert 0.85 < _R["auc_trap"] < 0.95, "有信号的打分器 AUC 应显著高于 0.5"


def test_f1_is_harmonic_mean():
    p, r, f1 = _R["precision"], _R["recall"], _R["f1"]
    assert abs(f1 - 2 * p * r / (p + r)) < 1e-12, "F1 应为 P、R 的调和平均"


def test_accuracy_misleads_on_imbalance():
    # 多数类基线：高准确率、零召回（一文不值）
    assert _R["acc_base"] > 0.98, "多数类基线在 1% 正类上准确率应≈99%"
    assert _R["rec_base"] == 0.0, "多数类基线召回应为 0（什么正例都没抓到）"
    # 真分类器：准确率甚至略低，召回却高得多
    assert _R["rec_clf"] > 0.8, f"真分类器召回应远高于基线，实际 {_R['rec_clf']:.3f}"
    assert _R["acc_clf"] <= _R["acc_base"] + 0.01, \
        "揭穿点：真正有用的分类器，准确率未必比无用基线高"


def test_kfold_cv_has_variance():
    accs = _R["cv_accs"]
    assert len(accs) == 5
    assert accs.std() > 1e-3, "k 折各折应有可见方差（说明单次留出不可靠）"
    assert 0.5 < accs.mean() < 0.95, "CV 均值应在合理区间"


def _run_all():
    passed = 0
    tests = [
        test_auc_two_algorithms_agree,
        test_f1_is_harmonic_mean,
        test_accuracy_misleads_on_imbalance,
        test_kfold_cv_has_variance,
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
