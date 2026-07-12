"""The AI Mind · 第 23 章配套代码：micro-autograd（工程里程碑①）

一台从零构建的微型自动微分引擎：Value 类记录数值、梯度、以及"如何反向传播"
（父节点引用 + _backward 闭包）；backward() 对计算图做拓扑排序，从输出（grad=1）
逆序回传，把梯度沿图分发（链式法则）。每条 _backward 里的 `+=` 是为了正确处理
fan-out（一个节点被多次使用时各路径梯度相加）。

两件事把里程碑跑成可核验的事实：
  - **梯度检验**：autograd 算的梯度 vs 有限差分数值梯度，相对误差极小（引擎正确）；
  - **训练 Ch22 的 MLP**：用这台引擎在 XOR 上跑梯度下降，损失下降、准确率达 1.0
    ——Ch22 搭好却训不了的网络，现在学会了。

这是 PyTorch/JAX autograd 与 Karpathy micrograd 的最小内核。纯 Python + numpy、固定种子。
运行 `python -m ch23.micrograd` 打印梯度检验与 XOR 训练结果。
"""
from __future__ import annotations

import math

import numpy as np


class Value:
    """包住一个标量的计算图节点。"""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad     # 上游 × 本地(1) → 分发（链式法则）
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad   # 乘法是"梯度交换器"
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad   # 本地梯度复用前向输出 t

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, o):
        return self * o ** -1

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        for v in topo:
            v.grad = 0.0
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def grad_check():
    """返回 (rel_a, rel_b)：autograd 梯度与有限差分的相对误差。"""
    a = Value(-4.0); b = Value(2.0)
    d = a * b + b ** 3
    e = ((a + b).tanh() + d) * a
    f = e * e / (a + 6.0)
    f.backward()
    ag, bg = a.grad, b.grad

    h = 1e-6

    def forward(av, bv):
        a = Value(av); b = Value(bv)
        d = a * b + b ** 3
        e = ((a + b).tanh() + d) * a
        return (e * e / (a + 6.0)).data

    num_a = (forward(-4.0 + h, 2.0) - forward(-4.0 - h, 2.0)) / (2 * h)
    num_b = (forward(-4.0, 2.0 + h) - forward(-4.0, 2.0 - h)) / (2 * h)
    rel_a = abs(ag - num_a) / (abs(num_a) + 1e-12)
    rel_b = abs(bg - num_b) / (abs(num_b) + 1e-12)
    return rel_a, rel_b


class Neuron:
    def __init__(self, nin, rng):
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, rng):
        self.neurons = [Neuron(nin, rng) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts, rng):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], rng) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def train_xor(seed=1, epochs=200, lr=0.1):
    """用 micro-autograd 训练一个 MLP 解 XOR，返回 (损失序列, 最终准确率)。"""
    rng = np.random.RandomState(seed)
    model = MLP(2, [4, 4, 1], rng)
    X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    Y = [-1., 1., 1., -1.]
    losses = []
    for _ in range(epochs):
        ypred = [model(x) for x in X]
        loss = sum(((yp - yt) ** 2 for yp, yt in zip(ypred, Y)), Value(0.0))
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        losses.append(loss.data)
    preds = [model(x).data for x in X]
    acc = sum((p > 0) == (t > 0) for p, t in zip(preds, Y)) / len(Y)
    return losses, float(acc)


def main() -> None:
    ra, rb = grad_check()
    print("grad a rel=%.2e   grad b rel=%.2e" % (ra, rb))
    print("---")
    losses, acc = train_xor()
    for ep in (0, 20, 40, 100, 199):
        print("epoch %3d loss %.6f" % (ep, losses[ep]))
    print("accuracy:", acc)


if __name__ == "__main__":
    main()
