"""第 26 章配套代码的回归测试。

卷积网络的核心事实（确定性，seed=0）：
  1. 从零训练到高准确率：手写反向传播的小 CNN 在"随机位置的横条 vs 竖条"上
     从随机初始化训练到 test_acc ≈ 0.97（远优于二分类基线 0.5），且训练/测试
     无过拟合缺口；
  2. 平移等变是可核验的事实：输入平移一列、特征图也平移一列，有效区域内最大
     逐元素差 = 0.0（数学精确，非巧合——卷积确定、核处处相同）；
  3. 卷积 ≪ 全连接的参数量：conv 40 vs 等价 FC 58000，比值 1450×（局部性 +
     权重共享把参数与图像尺寸解耦）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch26.convnet import run

_R = run()


def test_cnn_trains_from_scratch():
    assert _R["test_acc"] > 0.9, \
        f"小 CNN 应从零训练到高测试准确率，实际 {_R['test_acc']:.3f}"
    assert _R["train_acc"] > 0.9, \
        f"训练准确率也应高，实际 {_R['train_acc']:.3f}"


def test_no_overfitting_gap():
    # 训练/测试相当（差异落在 200 样本采样噪声内），无训练远高于测试的过拟合缺口
    assert _R["train_acc"] - _R["test_acc"] < 0.1, \
        f"不应出现过拟合缺口：train {_R['train_acc']:.3f} vs test {_R['test_acc']:.3f}"


def test_translation_equivariance_is_exact():
    # 平移等变在有效区域内数学精确成立
    assert _R["equiv_diff"] < 1e-9, \
        f"平移等变有效区域最大差应≈0，实际 {_R['equiv_diff']:.2e}"


def test_conv_params_far_fewer_than_fc():
    assert _R["conv_params"] == 40, f"卷积层参数应为 40，实际 {_R['conv_params']}"
    assert _R["fc_params"] == 58000, f"等价全连接参数应为 58000，实际 {_R['fc_params']}"
    assert _R["ratio"] > 1000, \
        f"卷积 vs 全连接参数比应达上千倍，实际 {_R['ratio']:.1f}"


def _run_all():
    passed = 0
    tests = [
        test_cnn_trains_from_scratch,
        test_no_overfitting_gap,
        test_translation_equivariance_is_exact,
        test_conv_params_far_fewer_than_fc,
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
