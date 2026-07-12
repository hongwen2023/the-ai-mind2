"""The AI Mind · 第 22 章配套代码：从感知机到多层网络

用 XOR 把本章的核心事实跑成可核验的结果：
  - **线性模型栽在 XOR**：逻辑回归梯度下降 2 万步仍只有 0.5 准确率（线性不可分）；
  - **手工构造的 2 层 ReLU MLP 解 XOR**：输出 [0,1,1,0]，准确率 1.0（本章不训练，
    直接给权重——反向传播留给 Ch23）；
  - **隐藏层把 XOR 变成线性可分**：两个正类在隐藏空间映到同一点，一个线性读出即可分开。

纯 NumPy、固定种子。
运行 `python -m ch22.perceptron_to_mlp` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([0., 1., 1., 0.])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0.0, z)


def train_linear_xor(seed=0, lr=0.5, steps=20000):
    """逻辑回归（线性）在 XOR 上做梯度下降，返回 (w, b, 预测, 准确率)。"""
    rng_state = np.random.RandomState(seed)
    w = rng_state.randn(2)
    b = 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        g = p - Y
        w -= lr * (X.T @ g) / len(Y)
        b -= lr * g.mean()
    pred = (sigmoid(X @ w + b) > 0.5).astype(float)
    return w, b, pred, float((pred == Y).mean())


def mlp_xor():
    """手工构造的 2 层 ReLU MLP 解 XOR，返回 (隐藏表示 H, 预测, 准确率)。"""
    W1 = np.array([[1., 1.], [1., 1.]])
    b1 = np.array([0., -1.])          # h1=relu(x1+x2), h2=relu(x1+x2-1)
    H = relu(X @ W1.T + b1)
    W2 = np.array([1., -2.]); b2 = 0.0  # 线性读出 h1-2h2
    pred = (H @ W2 + b2 > 0.5).astype(float)
    return H, pred, float((pred == Y).mean())


def hidden_is_linearly_separable(H, seed=0, lr=0.3, steps=5000):
    """在隐藏表示上训一个线性分类器；若能 100% 分开，则隐藏表示线性可分。"""
    rng = np.random.RandomState(seed)
    w = rng.randn(H.shape[1]) * 0.1
    b = 0.0
    for _ in range(steps):
        g = sigmoid(H @ w + b) - Y
        w -= lr * (H.T @ g) / len(Y)
        b -= lr * g.mean()
    pred = (sigmoid(H @ w + b) > 0.5).astype(float)
    return float((pred == Y).mean())


def main() -> None:
    w, b, pred_lin, acc_lin = train_linear_xor()
    print("线性模型 收敛 w =", np.round(w, 6), " b =", round(b, 6))
    print("线性模型预测:", pred_lin, " 准确率:", acc_lin)

    H, pred_mlp, acc_mlp = mlp_xor()
    print("MLP 输出预测:", pred_mlp, " 准确率:", acc_mlp)
    print("原始 -> 隐藏表示 (标签):")
    for xi, hi, yi in zip(X, H, Y):
        print(f"  {xi} -> {hi}   label={int(yi)}")
    print("隐藏表示上线性分类器准确率:", hidden_is_linearly_separable(H))


if __name__ == "__main__":
    main()
