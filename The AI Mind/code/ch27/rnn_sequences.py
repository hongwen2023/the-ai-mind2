"""The AI Mind · 第 27 章配套代码：从零实现 vanilla RNN + 手写 BPTT

把"为序列烤进的先验（时间上共享权重 + 隐藏状态压缩过去）"测成数字。
任务是**序列奇偶校验 (parity)**：输出一串比特的异或——要做对它，必须
(1) 跨整个序列维持"迄今为止的奇偶"这 1 bit 状态，(2) 做 XOR 这一非线性运算。

三块可断言的事实（纯 NumPy、固定种子）：
  1. 记忆与非线性缺一不可：两个逻辑回归稻草人（只看末位 / 看全序列但线性）
     都 ≈ 0.5；RNN（有记忆有非线性）在 T=8 上满分；
  2. 循环先验独占的结构红利：同一套 4353 个参数在 T=8/16/24 上**都有定义**，
     参数量与序列长度解耦（对照 Ch26 卷积参数与图像尺寸解耦）；无循环 MLP
     虽在固定 T=8 上也满分，却做不到"一机通吃所有长度"；
  3. vanilla RNN 的记忆天花板可测：直接在 T=16/24 上训练断崖跌回随机
     (≈0.5)，病因是梯度沿时间指数消失——回传到早期时间步的梯度范数比末步
     小约三个数量级，且解析线性化下严丝合缝地走 rho^(T-1)。

BPTT 手写（把 RNN 沿时间展开成前馈图，对共享权重按时间累加梯度，回指 Ch23）。
运行 `python -m ch27.rnn_sequences` 打印那组可核验输出（约 40–90 秒）。
"""
from __future__ import annotations

import numpy as np

D, H = 2, 64
N_train, N_test = 3000, 1000

# 模块级 RNG，由 run() 用固定种子重置；下方 helper 复用它以保证 RNG 调用顺序固定
rng = np.random.default_rng(0)


def make_data(n, T):
    # 输入：长度 T 的比特序列（每步 2 维 one-hot）；标签：全序列比特的异或 (XOR)。
    bits = rng.integers(0, 2, size=(n, T))
    y = bits.sum(axis=1) % 2
    X = np.zeros((n, T, D))
    X[np.arange(n)[:, None], np.arange(T)[None, :], bits] = 1.0
    return X, y.astype(np.float64), bits


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def init_params(seed=1):
    # 三组权重在所有时间步共享，与 T 无关
    r = np.random.default_rng(seed)
    Wxh = r.standard_normal((H, D)) * 0.5             # 输入 -> 隐藏
    Q, _ = np.linalg.qr(r.standard_normal((H, H)))    # 正交初始化，奇异值=谱半径=1
    Whh = Q                                           # 隐藏 -> 隐藏（循环）
    Why = r.standard_normal((1, H)) * 0.3             # 隐藏 -> 输出
    return dict(Wxh=Wxh, Whh=Whh, Why=Why, bh=np.zeros(H), by=np.zeros(1))


def forward(P, Xb):
    n, T, _ = Xb.shape
    h = np.zeros((n, H))
    hs = [h]
    for t in range(T):                                # 沿时间展开，每步复用同一份权重
        h = np.tanh(Xb[:, t, :] @ P['Wxh'].T + h @ P['Whh'].T + P['bh'])
        hs.append(h)
    p = sigmoid((hs[-1] @ P['Why'].T + P['by'])[:, 0])
    return p, hs


def evaluate(P, Xb, Yb):
    p, _ = forward(P, Xb)
    return float(np.mean((p > 0.5) == (Yb > 0.5)))


def train(P, Xtr, Ytr, T, lr=0.3, batch=128, epochs=150, clip=5.0):
    # 手写 BPTT：对共享权重按时间累加梯度
    for _ in range(epochs):
        idx = rng.permutation(len(Ytr))
        for s in range(0, len(Ytr), batch):
            bi = idx[s:s + batch]
            Xb, Yb = Xtr[bi], Ytr[bi]
            n = len(bi)
            h = np.zeros((n, H))
            hs = [h]
            for t in range(T):
                h = np.tanh(Xb[:, t, :] @ P['Wxh'].T + h @ P['Whh'].T + P['bh'])
                hs.append(h)
            p = sigmoid((hs[-1] @ P['Why'].T + P['by'])[:, 0])
            g = {k: np.zeros_like(v) for k, v in P.items()}
            dlogit = (p - Yb)[:, None] / n                 # 交叉熵+sigmoid 的干净梯度
            g['Why'] += dlogit.T @ hs[-1]
            g['by'] += dlogit.sum(0)
            dh = dlogit @ P['Why']                         # 流入 h_T 的梯度
            for t in reversed(range(T)):                   # 沿时间反传
                da = dh * (1 - hs[t + 1]**2)               # 过 tanh'
                g['Wxh'] += da.T @ Xb[:, t, :]             # 共享权重 -> 按时间累加
                g['Whh'] += da.T @ hs[t]                   # 共享权重 -> 按时间累加
                g['bh'] += da.sum(0)
                dh = da @ P['Whh']                         # 传给上一时间步
            for k in P:
                np.clip(g[k], -clip, clip, out=g[k])       # 逐元素按值裁剪：驯服爆炸
            for k in P:
                P[k] -= lr * g[k]
    return P


def train_mlp(Xf, y, Xf_te, y_te, hid=64, lr=0.3, epochs=300, batch=128):
    # 无循环 MLP，对展平的整段序列做二分类（固定长度下不需要循环）
    r = np.random.default_rng(2)
    d = Xf.shape[1]
    W1 = r.standard_normal((hid, d)) * np.sqrt(2 / d)
    b1 = np.zeros(hid)
    W2 = r.standard_normal(hid) * np.sqrt(2 / hid)
    b2 = 0.0
    for _ in range(epochs):
        idx = r.permutation(len(y))
        for s in range(0, len(y), batch):
            bi = idx[s:s + batch]
            xb = Xf[bi]
            yb = y[bi]
            n = len(bi)
            z1 = xb @ W1.T + b1
            a1 = np.maximum(z1, 0)
            p = sigmoid(a1 @ W2 + b2)
            dl = (p - yb) / n
            da1 = np.outer(dl, W2) * (z1 > 0)
            W1 -= lr * (da1.T @ xb)
            b1 -= lr * da1.sum(0)
            W2 -= lr * (a1.T @ dl)
            b2 -= lr * dl.sum()
    p = sigmoid(np.maximum(Xf_te @ W1.T + b1, 0) @ W2 + b2)
    return float(np.mean((p > 0.5) == (y_te > 0.5)))


def logreg(Xf, y, Xf_te, y_te, lr=0.5, epochs=300):
    d = Xf.shape[1]
    w = np.zeros(d)
    b = 0.0
    n = len(y)
    for _ in range(epochs):
        p = sigmoid(Xf @ w + b)
        gg = (p - y)
        w -= lr * (Xf.T @ gg) / n
        b -= lr * gg.mean()
    return float(np.mean((sigmoid(Xf_te @ w + b) > 0.5) == (y_te > 0.5)))


def grad_norms_over_time(P, Xb, Yb, T):
    # 在真实模型上：回传到各时间步的梯度范数
    n = len(Yb)
    h = np.zeros((n, H))
    hs = [h]
    for t in range(T):
        h = np.tanh(Xb[:, t, :] @ P['Wxh'].T + h @ P['Whh'].T + P['bh'])
        hs.append(h)
    p = sigmoid((hs[-1] @ P['Why'].T + P['by'])[:, 0])
    dh = ((p - Yb)[:, None] / n) @ P['Why']
    norms = []
    for t in reversed(range(T)):
        da = dh * (1 - hs[t + 1]**2)
        norms.append(float(np.linalg.norm(da)))
        dh = da @ P['Whh']
    return norms[::-1]                                  # 按时间步 1..T 排列


def jacobian_decay(rho, T=25, seed=1):
    # 解析对照：零输入线性化下 ||dh_T/dh_t|| = rho^k
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.standard_normal((H, H)))
    W = Q * rho                                         # 正交阵 => 奇异值=谱半径=rho
    h = np.zeros(H)
    hs = [h]
    for t in range(T):
        h = np.tanh(W @ h)                             # 零输入 => h 恒为 0，tanh'(0)=1
        hs.append(h)
    g = np.ones(H) / np.sqrt(H)
    norms = [1.0]
    for t in reversed(range(T)):
        g = W.T @ (g * (1 - hs[t + 1]**2))
        norms.append(float(np.linalg.norm(g)))
    return norms


def run(seed=0):
    global rng
    rng = np.random.default_rng(seed)

    T = 8
    Xtr, Ytr, Btr = make_data(N_train, T)
    Xte, Yte, Bte = make_data(N_test, T)
    P = init_params()
    n_params = int(sum(v.size for v in P.values()))
    train(P, Xtr, Ytr, T)
    rnn_acc = evaluate(P, Xte, Yte)

    mlp_acc = train_mlp(Xtr.reshape(N_train, -1), Ytr, Xte.reshape(N_test, -1), Yte)
    acc_last = logreg(Xtr[:, -1, :], Ytr, Xte[:, -1, :], Yte)
    acc_lin = logreg(Btr.astype(float), Ytr, Bte.astype(float), Yte)

    # 结构红利：同一套参数在任意长度上都有定义
    runs_at = {}
    for Tt in (8, 16, 24):
        Xg, _, _ = make_data(3, Tt)
        p, _ = forward(P, Xg)
        runs_at[Tt] = (bool(p.shape == (3,)), int(sum(v.size for v in P.values())))

    # 记忆天花板：直接在更长的 T 上训练 -> 断崖
    crash = {}
    for Tt in (8, 16, 24):
        Xa, Ya, _ = make_data(N_train, Tt)
        Xb2, Yb2, _ = make_data(N_test, Tt)
        crash[Tt] = evaluate(train(init_params(), Xa, Ya, Tt), Xb2, Yb2)

    # 病因：T=32 网络里回传到各时间步的梯度范数
    Xl, Yl, _ = make_data(512, 32)
    gn = grad_norms_over_time(init_params(), Xl, Yl, 32)

    dec = jacobian_decay(0.5)
    exp = jacobian_decay(1.5)

    return {
        "n_params": n_params,
        "rnn_acc": rnn_acc,
        "mlp_acc": mlp_acc,
        "acc_last": acc_last,
        "acc_lin": acc_lin,
        "runs_at": runs_at,
        "crash": crash,
        "gn": gn,
        "dec": dec,
        "exp": exp,
    }


def main() -> None:
    r = run()
    print("参数量 (与序列长度 T 无关):        %d" % r["n_params"])
    print("--- 固定长度 T=8 ---")
    print("RNN(循环) 准确率:                   %.3f" % r["rnn_acc"])
    print("MLP(展平, 无循环) 准确率:           %.3f" % r["mlp_acc"])
    print("无记忆基线(只看末位) 准确率:         %.3f" % r["acc_last"])
    print("线性基线(看全序列) 准确率:           %.3f" % r["acc_lin"])
    print("--- 结构红利: 同一套参数在任意长度都有定义 ---")
    for Tt in (8, 16, 24):
        print("  RNN 前向在 T=%2d 可运行: %s, 参数量仍为 %d" % (Tt, r["runs_at"][Tt][0], r["runs_at"][Tt][1]))
    print("--- 记忆天花板: 直接在长 T 上训练 vanilla RNN ---")
    for Tt in (8, 16, 24):
        print("  训练@T=%2d -> 测试准确率: %.3f" % (Tt, r["crash"][Tt]))
    print("--- 病因: T=32 网络里回传到各时间步的梯度范数 ---")
    print("  t = 1,  8, 16, 24, 32:", ["%.2e" % r["gn"][k] for k in (0, 7, 15, 23, 31)])
    print("--- 解析对照: 零输入线性化 ||dh_T/dh_t|| = rho^k ---")
    print("  rho=0.5, k=0,5,10,15,20:", [round(r["dec"][k], 6) for k in (0, 5, 10, 15, 20)])
    print("  rho=1.5, k=0,5,10,15,20:", [round(r["exp"][k], 3) for k in (0, 5, 10, 15, 20)])


if __name__ == "__main__":
    main()
