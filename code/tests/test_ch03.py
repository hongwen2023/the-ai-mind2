"""第 3 章配套代码的回归测试。

全部为确定性整数计数，跨环境稳定，可精确断言（对应正文那两张计数表）：
  - 朴素递归斐波那契调用数 = 2*fib(n+1) - 1（指数级）
  - 记忆化斐波那契调用数 = 2n - 1（线性）
  - 子集和暴力枚举检查的子集数 = 2^n

运行：pytest   或   python test_ch03.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch03.complexity_wall import fib_naive_calls, fib_memo_calls, subset_sum_bruteforce


def test_fib_naive_calls_exact():
    # 与正文表逐位一致
    assert fib_naive_calls(5) == 15
    assert fib_naive_calls(10) == 177
    assert fib_naive_calls(20) == 21891
    assert fib_naive_calls(30) == 2692537
    assert fib_naive_calls(35) == 29860703


def test_fib_naive_matches_closed_form():
    # 朴素调用数 = 2*fib(n+1) - 1
    def fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
    for n in range(0, 25):
        assert fib_naive_calls(n) == 2 * fib(n + 1) - 1


def test_fib_memo_calls_linear():
    # 记忆化调用数 = 2n - 1（n>=1）
    for n in range(1, 40):
        assert fib_memo_calls(n) == 2 * n - 1


def test_subset_sum_enumerates_2_to_the_n():
    for n in (5, 10, 15):
        nums = list(range(1, n + 1))
        _, checked = subset_sum_bruteforce(nums, 10 ** 9)  # 无解，强制全枚举
        assert checked == 2 ** n


def test_subset_sum_finds_existing_solution():
    found, _ = subset_sum_bruteforce([3, 34, 4, 12, 5, 2], 9)  # 4+5=9
    assert found is True


def _run_all():
    passed = 0
    tests = [
        test_fib_naive_calls_exact,
        test_fib_naive_matches_closed_form,
        test_fib_memo_calls_linear,
        test_subset_sum_enumerates_2_to_the_n,
        test_subset_sum_finds_existing_solution,
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
