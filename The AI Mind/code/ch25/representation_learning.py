"""The AI Mind · 第 25 章配套代码：表示学习

把"学到的表示让问题变简单"测成数字。数据是"藏进噪声的 XOR"：20 维里只有前 2 维
构成 Ch22 的 XOR（线性绝对不可分），后 18 维是纯噪声。

**线性探针 (linear probe)** + 三重基线，把"学习"的功劳从"升维"里单独逼出来：
  ① 原始输入上的线性探针        —— ≈ 随机（XOR 线性不可分）；
  ② 随机(未训练)隐藏表示上的探针 —— 也 ≈ 随机（只升维、没学，够不着信息子空间）；
  ③ MLP 学到的隐藏表示上的探针   —— 高准确率（表示把问题掰成线性可分）。
只有 ③ 显著赢过 ①②，才证明是"学习"而非"升维"让问题变简单。

外加嵌入几何演示（理想解耦表）：king-man+woman≈queen 的余弦类比。
纯 NumPy、固定种子。
运行 `python -m ch25.representation_learning` 打印那组可核验输出。
"""
from __future__ import annotations

import numpy as np

D = 20


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def make_xor(rng, n, c=0.8, blob=0.18):
    s = rng.integers(0, 2, (n, 2)) * 2 - 1
    y = (s[:, 0] * s[:, 1] < 0).astype(int)
    x01 = c * s + rng.normal(0, blob, (n, 2))
    noise = rng.normal(0, 1.0, (n, D - 2))
    return np.concatenate([x01, noise], 1), y


def train_linear(X, y, epochs=400, lr=0.5):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        g = sigmoid(X @ w + b) - y
        w -= lr * (X.T @ g) / len(X)
        b -= lr * g.mean()
    return w, b


def acc_linear(w, b, X, y):
    return float(((sigmoid(X @ w + b) > 0.5).astype(int) == y).mean())


def run(seed=0):
    rng = np.random.default_rng(seed)
    Xtr, ytr = make_xor(rng, 1500)
    Xte, yte = make_xor(rng, 600)

    # 基线①：原始输入
    w0, b0 = train_linear(Xtr, ytr)
    acc_raw = acc_linear(w0, b0, Xte, yte)

    # MLP: 20 -> 16 -> 1
    d_h = 16
    W1 = rng.normal(0, 0.5, (D, d_h)); b1 = np.zeros(d_h)
    W2 = rng.normal(0, 0.5, (d_h, 1)); b2 = np.zeros(1)

    # 基线②：未训练的随机投影（只升维+非线性，无学习）
    def project(X):
        return relu(X @ W1 + b1)
    wr, br = train_linear(project(Xtr), ytr)
    acc_rand = acc_linear(wr, br, project(Xte), yte)

    def forward(X):
        h = relu(X @ W1 + b1)
        o = sigmoid(h @ W2 + b2)[:, 0]
        return h, o

    lr = 0.2
    for _ in range(3000):
        h, o = forward(Xtr)
        n = len(Xtr)
        do = (o - ytr) / n
        gW2 = h.T @ do[:, None]; gb2 = do.sum()
        dh = (do[:, None] @ W2.T) * (h > 0)
        gW1 = Xtr.T @ dh; gb1 = dh.sum(0)
        W2 -= lr * gW2; b2 -= lr * gb2
        W1 -= lr * gW1; b1 -= lr * gb1

    _, o_te = forward(Xte)
    acc_mlp = float(((o_te > 0.5).astype(int) == yte).mean())

    # 基线③：学到的隐藏表示上的线性探针
    Htr, _ = forward(Xtr)
    Hte, _ = forward(Xte)
    wp, bp = train_linear(Htr, ytr)
    acc_probe = acc_linear(wp, bp, Hte, yte)

    # 嵌入几何（理想解耦表）
    emb = {"man": np.array([1.0, 0.0]), "woman": np.array([-1.0, 0.0]),
           "king": np.array([1.0, 2.0]), "queen": np.array([-1.0, 2.0])}

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    analogy = emb["king"] - emb["man"] + emb["woman"]
    return {
        "acc_raw": acc_raw, "acc_rand": acc_rand,
        "acc_mlp": acc_mlp, "acc_probe": acc_probe,
        "cos_analogy_queen": cos(analogy, emb["queen"]),
        "cos_king_queen": cos(emb["king"], emb["queen"]),
        "cos_king_woman": cos(emb["king"], emb["woman"]),
    }


def main() -> None:
    r = run()
    print(f"线性探针 @ 原始输入          : {r['acc_raw']:.3f}")
    print(f"线性探针 @ 随机(未训练)表示  : {r['acc_rand']:.3f}")
    print(f"MLP 端到端准确率             : {r['acc_mlp']:.3f}")
    print(f"线性探针 @ 学到的隐藏表示    : {r['acc_probe']:.3f}")
    print(f"king-man+woman vs queen 余弦 : {r['cos_analogy_queen']:.3f}")
    print(f"king vs queen 余弦            : {r['cos_king_queen']:.3f}")
    print(f"king vs woman 余弦            : {r['cos_king_woman']:.3f}")


if __name__ == "__main__":
    main()
