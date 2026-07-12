"""The AI Mind · 第 29 章配套代码：从零实现注意力（Book V 开篇）

把「注意力 = 可微分的按内容软查找」测成数字。纯 NumPy、固定种子。

两块可断言的事实（seed=0）：
  1. 按内容检索 (lookup)：6 个记忆槽，带噪声的 query 能让注意力权重集中到
     正确的槽（准确率 1.0、干净查询下正确槽拿走 ≈0.916 的注意力预算），
     取回正确 value（与真值 L2 ≈0.185，远小于 V 的典型幅度 ≈1）——
     每行权重非负且和为 1，是合法概率分布；
  2. 缩放的必要：对纯随机 query/key、每个维度平均 2000 次——不除 √d_k 时
     softmax 随维度单调变尖（d_k=512 最大权重 ≈0.937、熵塌到 ≈0.157，朝
     one-hot 逼近、梯度会死），除 √d_k 则几乎对维度免疫（最大权重稳在 ≈0.16、
     熵稳在 ≈3.0）。

连回概念：点积=相似度 (Ch8)；softmax 权重=概率分布 (Ch12)；数值稳定 softmax
(Ch9)；√d_k 缩放把 q·k 的方差从 d_k 拉回 O(1)。

运行 `python -m ch29.attention` 打印那组可核验输出。
"""
from __future__ import annotations

import numpy as np


def softmax(x, axis=-1):
    # 减去每行最大值再取 exp：数学等价但避免溢出（Ch12/Ch9 数值稳定）
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(Q, K, V, scale=True):
    # 缩放点积注意力：softmax(Q K^T / sqrt(d_k)) V
    d_k = Q.shape[-1]
    scores = Q @ K.T                     # (m,n)：每个查询对每个键的点积分数表
    if scale:
        scores = scores / np.sqrt(d_k)   # 除以 sqrt(d_k)，把方差拉回 O(1)
    A = softmax(scores, axis=-1)         # (m,n)：每行是一个概率分布（注意力权重）
    return A @ V, A                      # 输出 (m,d_v)，外加权重矩阵 A 供检视


def entropy(p):
    # 熵越小=分布越尖（越接近 one-hot）；越大=越平摊
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def run(seed=0):
    rng = np.random.default_rng(seed)

    # ---- (1) 按内容检索 lookup ----
    n_slots, d_k, d_v = 6, 16, 4
    M = rng.standard_normal((n_slots, d_k))
    Qo, _ = np.linalg.qr(M.T)
    K = Qo.T[:n_slots] * np.sqrt(d_k)        # 6 个互相正交、区分度高的地址(key)
    V = rng.standard_normal((n_slots, d_v))  # 6 段内容(value)

    correct, maxw, trials = 0, [], 500
    A = None
    for _ in range(trials):
        j = rng.integers(n_slots)                     # 随机目标槽
        q = K[j] + 0.5 * rng.standard_normal(d_k)     # 带噪声的查询
        _, A = attention(q[None], K, V, scale=True)
        correct += int(np.argmax(A[0]) == j)
        maxw.append(A[0].max())

    row_sum = float(A.sum(1)[0])
    accuracy = correct / trials
    avg_max_weight = float(np.mean(maxw))
    o, Aclean = attention(K[2][None], K, V, scale=True)   # 干净地址 #2 精确查一次
    clean_weights = np.round(Aclean[0], 3)
    retrieved_l2 = float(np.linalg.norm(o[0] - V[2]))
    var_V = float(np.var(V))

    # ---- (2) 缩放对照 ----
    n_keys, n_samples = 32, 2000
    scaling = {}
    for dk in [8, 64, 512]:
        sm_s, se_s, sm_u, se_u = [], [], [], []
        for _ in range(n_samples):
            Kr = rng.standard_normal((n_keys, dk))
            qr = rng.standard_normal(dk)
            _, As = attention(qr[None], Kr, np.zeros((n_keys, 1)), scale=True)
            _, Au = attention(qr[None], Kr, np.zeros((n_keys, 1)), scale=False)
            sm_s.append(As[0].max()); se_s.append(entropy(As[0]))
            sm_u.append(Au[0].max()); se_u.append(entropy(Au[0]))
        scaling[dk] = {
            "scaled_maxw": float(np.mean(sm_s)), "scaled_H": float(np.mean(se_s)),
            "unscaled_maxw": float(np.mean(sm_u)), "unscaled_H": float(np.mean(se_u)),
        }

    return {
        "row_sum": row_sum, "accuracy": accuracy, "avg_max_weight": avg_max_weight,
        "clean_weights": clean_weights, "retrieved_l2": retrieved_l2, "var_V": var_V,
        "scaling": scaling,
    }


def main() -> None:
    r = run()
    print("[lookup] row-sum of A:", round(r["row_sum"], 6))
    print("[lookup] accuracy:", r["accuracy"])
    print("[lookup] avg max weight:", round(r["avg_max_weight"], 3))
    print("[lookup] weights (clean q=addr#2):", r["clean_weights"])
    print("[lookup] retrieved vs true V, L2:", round(r["retrieved_l2"], 3))
    print("[lookup] sample var of V:", round(r["var_V"], 3))
    print("\n[scaling] iid query/keys, averaged over many samples:")
    for dk in [8, 64, 512]:
        s = r["scaling"][dk]
        print(f"  d_k={dk:4d}: scaled maxw={s['scaled_maxw']:.3f} H={s['scaled_H']:.3f}"
              f" | unscaled maxw={s['unscaled_maxw']:.3f} H={s['unscaled_H']:.3f}")


if __name__ == "__main__":
    main()
