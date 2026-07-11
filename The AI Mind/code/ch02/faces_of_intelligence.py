"""The AI Mind · 第 2 章配套代码：智能的多副面孔（搜索这一注）

把第 2 章工程线的井字棋 minimax 实验沉淀为可导入、可测试的模块。

核心：用 minimax（含 alpha-beta 剪枝）完全求解井字棋，证明搜索在**可穷举的小世界**里
能下出证明意义上不败的完美棋；用节点计数把"搜索的代价"量化，并与无剪枝版本对照，
亲眼看到剪枝作为"注入归纳偏置"的威力——以及这一切在 b^d 组合爆炸面前如何注定撞墙。

约定：X 为极大方(先手)，O 为极小方(后手)。空格=0，X=1，O=-1。

运行 `python -m ch02.faces_of_intelligence` 打印正文那组可核验输出。
"""
from __future__ import annotations

import math

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 行
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 列
    (0, 4, 8), (2, 4, 6),             # 对角线
]


def winner(board):
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


def full(board):
    return all(v != 0 for v in board)


def minimax(board, player, counter, alpha=-2, beta=2, prune=True):
    """极大极小搜索。counter[0] 累计被展开的节点数；prune=False 关闭 alpha-beta。"""
    counter[0] += 1
    w = winner(board)
    if w != 0:
        return w
    if full(board):
        return 0

    if player == 1:  # X 走：极大化
        best = -2
        for i in range(9):
            if board[i] == 0:
                board[i] = 1
                val = minimax(board, -1, counter, alpha, beta, prune)
                board[i] = 0
                if val > best:
                    best = val
                if prune:
                    if best > alpha:
                        alpha = best
                    if alpha >= beta:
                        break
        return best
    else:            # O 走：极小化
        best = 2
        for i in range(9):
            if board[i] == 0:
                board[i] = -1
                val = minimax(board, 1, counter, alpha, beta, prune)
                board[i] = 0
                if val < best:
                    best = val
                if prune:
                    if best < beta:
                        beta = best
                    if alpha >= beta:
                        break
        return best


def best_move(board, player, counter):
    """返回当前玩家的最优落子位置与其博弈价值。"""
    best_val = -2 if player == 1 else 2
    best_i = -1
    for i in range(9):
        if board[i] == 0:
            board[i] = player
            val = minimax(board, -player, counter)
            board[i] = 0
            if player == 1 and val > best_val:
                best_val, best_i = val, i
            if player == -1 and val < best_val:
                best_val, best_i = val, i
    return best_i, best_val


def selfplay():
    """两个完美玩家对弈，返回 (终局结果, 总展开节点数, 终局棋盘)。"""
    board = [0] * 9
    counter = [0]
    player = 1
    while winner(board) == 0 and not full(board):
        i, _ = best_move(board, player, counter)
        board[i] = player
        player = -player
    return winner(board), counter[0], board


def solve_from_empty(prune=True):
    """从空棋盘完整求解，返回 (博弈价值, 展开节点数)。"""
    counter = [0]
    val = minimax([0] * 9, 1, counter, prune=prune)
    return val, counter[0]


def main() -> None:
    val, nodes = solve_from_empty(prune=True)
    print("空棋盘博弈价值(X先手):", val, "=> 0 表示双方完美时必和")
    print("完整求解展开的节点数(含alpha-beta剪枝):", nodes)

    result, sp_nodes, _ = selfplay()
    print("完美对弈结果:", {1: "X胜", -1: "O胜", 0: "平局"}[result])

    _, nodes_np = solve_from_empty(prune=False)
    print("无剪枝完整求解展开节点数:", nodes_np)
    print(f"剪枝加速比: {nodes_np}/{nodes} = {nodes_np / nodes:.2f}")

    print("叶子数上界 9! =", math.factorial(9))


if __name__ == "__main__":
    main()
