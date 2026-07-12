"""The AI Mind · 第 26 章配套代码：从零训练一个小 CNN（工程里程碑②）

把"图像的两条先验（局部性 + 平移不变）烤进架构"测成数字。纯 NumPy、
固定种子 default_rng(0)、手写卷积/池化的前向与反向，从随机初始化训练一个
"卷积 → ReLU → 2x2 最大池化 → 线性读出"的小 CNN，区分**随机位置**上的
横条 vs 竖条（随机位置正是为了逼出平移不变的价值）。

三条可断言的性质（seed=0，逐条可复现）：
  1. 从零训练到高准确率：test_acc ≈ 0.97（二分类基线 0.5）；
  2. 平移等变是可核验的事实：把输入平移一列、把特征图也平移一列，
     有效区域内最大逐元素差 = 0.0（卷积确定、核处处相同）；
  3. 卷积 ≪ 全连接的参数量：conv 40 vs 等价 FC 58000，比值 1450×。

反向传播全部手写，直接兑现 Ch23：卷积的反向是把上游梯度按同一滑窗规则
散射回权重与输入——dW 是"上游梯度 × 对应 patch"在所有位置的累加，正是
"权重共享 → 所有位置梯度累加到同一套核"。

运行 `python -m ch26.convnet` 打印那组可核验输出。
"""
from __future__ import annotations

import numpy as np

IMG = 12
BAR = 4  # 条的长度


def make_sample(rng, kind):
    x = np.zeros((IMG, IMG), dtype=np.float64)
    if kind == 0:  # 横条
        r = rng.integers(0, IMG)
        c = rng.integers(0, IMG - BAR + 1)
        x[r, c:c + BAR] = 1.0
    else:          # 竖条
        r = rng.integers(0, IMG - BAR + 1)
        c = rng.integers(0, IMG)
        x[r:r + BAR, c] = 1.0
    x += 0.05 * rng.standard_normal((IMG, IMG))  # 一点噪声
    return x


def make_dataset(rng, n):
    X = np.zeros((n, 1, IMG, IMG))
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        k = rng.integers(0, 2)
        X[i, 0] = make_sample(rng, k)
        y[i] = k
    return X, y


def conv_forward(x, W, b):
    # x:(N,Cin,H,W)  W:(Cout,Cin,kh,kw)  b:(Cout,)
    N, Cin, H, Wd = x.shape
    Cout, _, kh, kw = W.shape
    Ho, Wo = H - kh + 1, Wd - kw + 1
    out = np.zeros((N, Cout, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            patch = x[:, :, i:i + kh, j:j + kw]                # (N,Cin,kh,kw)
            out[:, :, i, j] = np.tensordot(patch, W, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


def conv_backward(dout, x, W):
    N, Cin, H, Wd = x.shape
    Cout, _, kh, kw = W.shape
    Ho, Wo = dout.shape[2], dout.shape[3]
    dx = np.zeros_like(x)
    dW = np.zeros_like(W)
    db = dout.sum(axis=(0, 2, 3))
    for i in range(Ho):
        for j in range(Wo):
            patch = x[:, :, i:i + kh, j:j + kw]                # (N,Cin,kh,kw)
            g = dout[:, :, i, j]                               # (N,Cout)
            dW += np.tensordot(g, patch, axes=([0], [0]))      # 权重共享→所有位置梯度累加
            dx[:, :, i:i + kh, j:j + kw] += np.tensordot(g, W, axes=([1], [0]))
    return dx, dW, db


def maxpool_forward(x):
    N, C, H, Wd = x.shape
    Ho, Wo = H // 2, Wd // 2
    out = np.zeros((N, C, Ho, Wo))
    mask = np.zeros_like(x)
    for i in range(Ho):
        for j in range(Wo):
            region = x[:, :, 2 * i:2 * i + 2, 2 * j:2 * j + 2].reshape(N, C, 4)
            idx = region.argmax(axis=2)
            out[:, :, i, j] = region.max(axis=2)
            for a in range(2):
                for bb in range(2):
                    mask[:, :, 2 * i + a, 2 * j + bb] = (idx == (a * 2 + bb))
    return out, mask


def maxpool_backward(dout, mask):
    N, C, Ho, Wo = dout.shape
    dx = np.zeros_like(mask)
    for i in range(Ho):
        for j in range(Wo):
            for a in range(2):
                for bb in range(2):
                    dx[:, :, 2 * i + a, 2 * j + bb] = mask[:, :, 2 * i + a, 2 * j + bb] * dout[:, :, i, j]
    return dx


def relu(x):
    return np.maximum(0, x)


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(p[np.arange(N), y] + 1e-12).mean()
    d = p.copy()
    d[np.arange(N), y] -= 1
    d /= N
    return loss, d


def run(seed=0):
    rng = np.random.default_rng(seed)

    # 数据（先于权重初始化，保证 RNG 调用顺序固定）
    Xtr, ytr = make_dataset(rng, 400)
    Xte, yte = make_dataset(rng, 200)

    # 初始化：conv 12->10, pool 10->5, flatten 4*5*5=100 -> 2 类
    Cout, k = 4, 3
    W1 = rng.standard_normal((Cout, 1, k, k)) * 0.3
    b1 = np.zeros(Cout)
    FLAT = Cout * 5 * 5
    W2 = rng.standard_normal((FLAT, 2)) * 0.1
    b2 = np.zeros(2)

    def forward(x):
        c = conv_forward(x, W1, b1)
        a = relu(c)
        p, mask = maxpool_forward(a)
        flat = p.reshape(x.shape[0], -1)
        logits = flat @ W2 + b2
        return logits, (x, c, a, p, mask, flat)

    def accuracy(X, y):
        logits, _ = forward(X)
        return float((logits.argmax(1) == y).mean())

    lr = 0.05
    curve = []
    for epoch in range(30):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 32):
            idx = perm[s:s + 32]
            xb, yb = Xtr[idx], ytr[idx]
            logits, cache = forward(xb)
            loss, dlog = softmax_ce(logits, yb)
            x, c, a, p, mask, flat = cache
            dW2 = flat.T @ dlog
            db2 = dlog.sum(0)
            dp = (dlog @ W2.T).reshape(p.shape)
            da = maxpool_backward(dp, mask)
            dc = da * (c > 0)                          # ReLU 反向
            _, dW1, db1 = conv_backward(dc, x, W1)
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1
        if epoch % 5 == 0 or epoch == 29:
            curve.append((epoch, float(loss), accuracy(Xtr, ytr), accuracy(Xte, yte)))

    train_acc = accuracy(Xtr, ytr)
    test_acc = accuracy(Xte, yte)

    # 平移等变数值验证
    xs = Xte[:1].copy()
    c0 = conv_forward(xs, W1, b1)
    xs_shift = np.roll(xs, shift=1, axis=3)            # 输入向右平移一列
    c1 = conv_forward(xs_shift, W1, b1)
    c0_shift = np.roll(c0, shift=1, axis=3)            # 把原特征图也平移一列
    equiv_diff = float(np.abs(c1[:, :, :, 1:] - c0_shift[:, :, :, 1:]).max())  # 忽略环绕边界列

    # 参数量对比
    conv_params = int(W1.size + b1.size)
    fc_hidden = Cout * 10 * 10                         # 等价全连接隐藏单元数
    fc_params = int(144 * fc_hidden + fc_hidden)

    return {
        "curve": curve,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "equiv_diff": equiv_diff,
        "conv_params": conv_params,
        "fc_params": fc_params,
        "ratio": fc_params / conv_params,
    }


def main() -> None:
    r = run()
    for epoch, loss, tr, te in r["curve"]:
        print(f"epoch {epoch:2d}  loss {loss:.4f}  train_acc {tr:.3f}  test_acc {te:.3f}")
    print("FINAL train_acc", round(r["train_acc"], 3), "test_acc", round(r["test_acc"], 3))
    print("equivariance max diff (valid region):", round(r["equiv_diff"], 8))
    print("conv layer params:", r["conv_params"], " equivalent FC params:", r["fc_params"],
          " ratio:", round(r["ratio"], 1))


if __name__ == "__main__":
    main()
