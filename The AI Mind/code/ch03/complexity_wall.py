"""The AI Mind · 第 3 章配套代码：计算的能力与代价

让"多项式 vs 指数"的质的鸿沟在屏幕上长出来。纯标准库、确定性。
证据以**调用/操作计数**为准（整数、跨机器一致、可被回归测试逐位断言）；
计时随机器晃动，不作为证据，故本模块不打印也不断言计时。

三个实验：
  A. 同一个问题（斐波那契），两种算法——朴素递归 O(φ^n) vs 记忆化 O(n)，数函数调用次数。
  B. 一个 NP 风格的暴力搜索——子集和 (subset-sum) 枚举全部 2^n 个子集。
  C. 把 n^2 与 2^n 并排摆出来。

运行 `python -m ch03.complexity_wall` 打印正文那组可核验的计数表。
"""
from __future__ import annotations


# ---------- 实验 A：同一个问题，两种算法 ----------
def fib_naive(n, counter):
    """朴素递归斐波那契。counter[0] 累计函数调用次数（指数级）。"""
    counter[0] += 1
    if n < 2:
        return n
    return fib_naive(n - 1, counter) + fib_naive(n - 2, counter)


def fib_memo(n, counter, memo=None):
    """记忆化斐波那契。counter[0] 累计函数调用次数（线性：2n-1）。"""
    if memo is None:
        memo = {}
    counter[0] += 1
    if n < 2:
        return n
    if n in memo:
        return memo[n]
    memo[n] = fib_memo(n - 1, counter, memo) + fib_memo(n - 2, counter, memo)
    return memo[n]


def fib_naive_calls(n):
    c = [0]
    fib_naive(n, c)
    return c[0]


def fib_memo_calls(n):
    c = [0]
    fib_memo(n, c)
    return c[0]


# ---------- 实验 B：一个 NP 风格的暴力搜索（子集和） ----------
def subset_sum_bruteforce(nums, target):
    """枚举全部 2^n 个子集，返回 (是否存在解, 检查的子集数)。"""
    n = len(nums)
    checked = 0
    for mask in range(1 << n):          # 1<<n 就是 2^n
        checked += 1
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s == target:
            return True, checked
    return False, checked


def main() -> None:
    print("=== 实验 A：斐波那契调用数 ===")
    print(f"{'n':>4} | {'朴素递归调用':>14} | {'记忆化调用':>12}")
    for n in (5, 10, 20, 30, 35):
        print(f"{n:>4} | {fib_naive_calls(n):>14} | {fib_memo_calls(n):>12}")

    print("\n=== 实验 B：子集和枚举的子集数 ===")
    print(f"{'n':>4} | {'检查的子集数':>14}")
    for n in (10, 20):
        nums = list(range(1, n + 1))
        _, checked = subset_sum_bruteforce(nums, 999999)  # 故意无解，强制全枚举
        print(f"{n:>4} | {checked:>14}")

    print("\n=== 多项式 vs 指数：把数摆出来 ===")
    print(f"{'n':>4} | {'n^2':>6} | {'2^n':>22}")
    for n in (10, 20, 30, 40, 50):
        print(f"{n:>4} | {n * n:>6} | {2 ** n:>22}")


if __name__ == "__main__":
    main()
