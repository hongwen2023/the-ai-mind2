"""第 27 章配套代码的回归测试。

循环网络的核心事实（seed=0）。断言按正文"复现口径"分两档：
  - 纯确定性代数（参数量 4353、rho^k 序列）做**精确**断言；
  - 依赖训练动力学/BLAS 的准确率与梯度范数做**量级/趋势**断言（贴 1.0 / 贴 0.5 /
    指数衰减），不锁末位数字。

三块结论：
  1. 记忆 + 非线性缺一不可：两个逻辑回归稻草人 ≈ 0.5，RNN(T=8) 满分；
  2. 循环先验独占结构红利：同一套 4353 参数在 T=8/16/24 都有定义（与 T 解耦）；
  3. vanilla RNN 记忆天花板：T=16/24 训练断崖回随机；病因是梯度沿时间指数消失
     （回传范数早步 << 末步；解析线性化严丝合缝走 rho^(T-1)）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch27.rnn_sequences import run

_R = run()


def test_param_count_is_length_independent():
    # 纯确定性：D=2,H=64,C=1 -> 64*2 + 64*64 + 1*64 + 64 + 1 = 4353
    assert _R["n_params"] == 4353, f"参数量应为 4353，实际 {_R['n_params']}"
    for Tt in (8, 16, 24):
        ok, nparams = _R["runs_at"][Tt]
        assert ok, f"RNN 前向应在 T={Tt} 可运行（一机通吃所有长度）"
        assert nparams == 4353, f"参数量应与 T 无关（仍 4353），T={Tt} 实际 {nparams}"


def test_rnn_solves_parity_baselines_fail():
    # RNN 有记忆有非线性 -> 满分；两个稻草人各缺一个条件 -> ≈随机
    assert _R["rnn_acc"] > 0.95, f"RNN 应解出 T=8 parity，实际 {_R['rnn_acc']:.3f}"
    assert _R["mlp_acc"] > 0.95, f"展平 MLP 在固定 T=8 也应满分，实际 {_R['mlp_acc']:.3f}"
    assert _R["acc_last"] < 0.6, f"只看末位比特应≈随机，实际 {_R['acc_last']:.3f}"
    assert _R["acc_lin"] < 0.6, f"线性看全序列应≈随机（XOR 不可线性分割），实际 {_R['acc_lin']:.3f}"


def test_memory_ceiling_crash_at_longer_T():
    # T=8 满分，但直接在更长 T 训练 vanilla RNN 断崖回随机
    assert _R["crash"][8] > 0.95, f"T=8 应满分，实际 {_R['crash'][8]:.3f}"
    assert _R["crash"][16] < 0.7, f"T=16 应断崖回随机，实际 {_R['crash'][16]:.3f}"
    assert _R["crash"][24] < 0.7, f"T=24 应断崖回随机，实际 {_R['crash'][24]:.3f}"


def test_gradient_vanishes_over_time():
    # 真实网络：回传到早期时间步的梯度范数 << 末步（指数消失可见）
    gn = _R["gn"]
    assert gn[0] < gn[-1] * 0.01, \
        f"早步梯度应比末步小两个数量级以上：t1={gn[0]:.2e} vs t32={gn[-1]:.2e}"


def test_jacobian_decay_is_exact_power_law():
    # 纯确定性代数：正交初始化下 ||dh_T/dh_t|| = rho^k 精确成立
    dec, exp = _R["dec"], _R["exp"]
    for k in (0, 5, 10, 15, 20):
        assert abs(dec[k] - 0.5 ** k) < 1e-6, f"rho=0.5 第 {k} 步应=0.5^{k}，实际 {dec[k]}"
        assert abs(exp[k] - 1.5 ** k) < 1e-2 * (1.5 ** k), f"rho=1.5 第 {k} 步应=1.5^{k}，实际 {exp[k]}"


def _run_all():
    passed = 0
    tests = [
        test_param_count_is_length_independent,
        test_rnn_solves_parity_baselines_fail,
        test_memory_ceiling_crash_at_longer_T,
        test_gradient_vanishes_over_time,
        test_jacobian_decay_is_exact_power_law,
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
