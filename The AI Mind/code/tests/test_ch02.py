"""第 2 章配套代码的回归测试。

这些数值是**确定性**的（无随机、纯整数搜索），跨环境稳定，因此可以精确断言：
  - 空棋盘博弈价值 = 0（井字棋双方完美时必和）
  - 完美自我对弈结果 = 平局
  - 含 alpha-beta 剪枝完整求解展开 18297 个节点
  - 无剪枝完整求解展开 549946 个节点，加速比 ≈ 30.06（与正文一致）

运行：
  pytest
  python test_ch02.py   # 无 pytest 时的兜底自检
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch02.faces_of_intelligence import solve_from_empty, selfplay


def test_game_value_is_draw():
    val, _ = solve_from_empty(prune=True)
    assert val == 0, f"井字棋空棋盘博弈价值应为 0（必和），实际 {val}"


def test_perfect_selfplay_is_draw():
    result, _, _ = selfplay()
    assert result == 0, f"两完美玩家对弈应平局，实际 {result}"


def test_pruned_node_count_exact():
    _, nodes = solve_from_empty(prune=True)
    assert nodes == 18297, f"含剪枝求解节点数应为 18297，实际 {nodes}"


def test_unpruned_node_count_exact():
    _, nodes = solve_from_empty(prune=False)
    assert nodes == 549946, f"无剪枝求解节点数应为 549946，实际 {nodes}"


def test_pruning_speedup_ratio():
    _, pruned = solve_from_empty(prune=True)
    _, unpruned = solve_from_empty(prune=False)
    ratio = unpruned / pruned
    assert abs(ratio - 30.06) < 0.05, f"剪枝加速比应约 30.06，实际 {ratio:.2f}"


def _run_all():
    passed = 0
    tests = [
        test_game_value_is_draw,
        test_perfect_selfplay_is_draw,
        test_pruned_node_count_exact,
        test_unpruned_node_count_exact,
        test_pruning_speedup_ratio,
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
