# 第 23 章 · 反向传播与自动微分

## ① 概念主线：从"存在权重"到"找到权重"

上一章结束在一个悬而未决的胜利上。万能逼近定理 (universal approximation theorem, UAT) 告诉我们：只要隐藏层足够宽，一个 MLP 就能把 XOR、把螺旋、把任何连续函数逼近到你想要的精度。存在这样一组权重。我们甚至亲手搭好了那个网络——两个输入、几个隐藏神经元、一个输出，前向传播的每一行代码都写清楚了。

然后我们停在了那里，没有训练它。

因为"存在"和"找到"之间隔着一条鸿沟。UAT 是一个存在性定理：它承诺在几百万维的权重空间里，某个点能让网络做对。它没有告诉你那个点在哪，也没有给你一张去往那里的地图。而学习的全部难处，恰恰在这句"在哪"。学习是"从经验中获得达成目标的能力"——具体到一个网络的毫米级动作，就是这一句：**根据数据，把权重从一个随机的、什么都不会的点，挪到一个能做对的点**。（中心对象的完整定义，我们留到本章末尾再郑重取出。）

我们在 Ch14 已经有了挪动的机制：梯度下降。站在损失曲面上的任意一点，梯度指向最陡上升方向，沿它的反方向走一小步，损失就下降一点。反复走，直到谷底。这套机制不挑维度——两维能走，两百万维也能走，数学上一模一样。

问题只剩一个，但它是全部：**你得先算出梯度**。

一个有两百万参数的网络，它的损失是这两百万个数的函数，梯度就是两百万个偏导数 $\partial L/\partial w_i$。手推？你连一个三层网络的梯度都推不干净——链式法则会在你笔下爆炸成一团分不清项的乘积。就算你推出来了，换个架构、加一层、把 tanh 换成 ReLU，你得从头再推一遍。这条路走不通。

所以本章要造一台机器。这台机器接受任意一张计算图——加、乘、激活函数随意组合——自动、精确、高效地吐出每一个参数的梯度。它的名字叫**反向传播 (backpropagation)**，而它背后的通用技术叫**反向模式自动微分 (reverse-mode automatic differentiation)**。

这里有一句话你要记一辈子：**反向传播不是新数学**。它没有超出 Ch9 的链式法则半步。Ch9 我们讲过一个几何洞见——链式法则就是雅可比矩阵相乘，而当输出端是一个标量时，从标量端起手做乘法、按乘法结合律从右往左结合，是所有结合顺序里最省的那个。那一章我们把这个洞见讲透了，但它停留在纸面。本章做的唯一一件事，是把那个洞见**机械化**：变成一个能在计算图上自动跑起来的算法，变成一百多行能真正训练网络的 Python 代码。

Ch22 搭好却训不了的那个网络，到本章结束时，会学会 XOR。这是本书工程主脊的第一个里程碑。

## ② 数学线（The Why）：计算图上的逆向流动

### 一切都是计算图

先把对象看清楚。任何一个表达式——无论是 $L = (\sigma(wx+b) - y)^2$ 还是一个五层网络的损失——都可以拆成一串**基本运算**：两个数相加、两个数相乘、一个数过一次激活函数。把每个中间结果画成一个节点，把"谁是谁的输入"画成有向边，你就得到一张**计算图 (computation graph)**。它一定是有向无环的 (directed acyclic graph, DAG)，因为计算有明确的先后，不会绕回去。

举个能捏在手里的例子。设 $z = x \cdot y$，再 $L = \tanh(z)$。这张图三个节点：$x$ 和 $y$ 是叶子，$z$ 由乘法产生，$L$ 由 tanh 产生。**前向传播 (forward pass)** 就是沿边的方向、按拓扑序把每个节点的数值算出来：先有 $x, y$，才能算 $z$，才能算 $L$。这一步平淡无奇，就是"求值"。

有意思的是反过来。

### 反向模式：从标量端起手，逆流而上

我们要的是 $\partial L/\partial x$ 和 $\partial L/\partial y$。链式法则说：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}\cdot\frac{\partial z}{\partial x}.$$

而 $\partial L/\partial z$ 自己也由链式法则给出：$L = \tanh(z)$，这一步的**本地导数**是 $\partial L/\partial z = 1 - \tanh^2 z$；它上游的梯度是 $\partial L/\partial L = 1$（损失对自己的导数）。两者相乘：

$$\frac{\partial L}{\partial z} = \underbrace{1}_{\text{上游}}\cdot\underbrace{(1-\tanh^2 z)}_{\text{本地}} = 1-\tanh^2 z.$$

盯住这个结构。每个节点手里都握着一个数——它对最终损失的偏导 $\partial L/\partial(\text{本节点})$，我们叫它这个节点的**梯度 (gradient)**，记作该节点的 `grad`。反向传播的核心动作，是让梯度**沿图逆向流动**，一个节点一个节点地传：

> 每个节点收到从**上游**（离损失更近的一侧）传来的梯度 $\partial L/\partial(\text{本节点输出})$，把它乘上**本地梯度**（这个运算的输出对它某个输入的偏导 $\partial(\text{输出})/\partial(\text{输入})$），得到的乘积就是要**分发**给那个输入的梯度。

这就是链式法则，一字不差。"上游梯度 × 本地梯度 = 分发给输入的梯度"——这一句是本章的心脏，后面每写一行 `_backward` 代码，你都要在心里默念它。

三个基本运算的本地梯度，我们必须手上有，因为它们是引擎的原子：

- **加法** $z = x + y$：$\dfrac{\partial z}{\partial x} = 1,\ \dfrac{\partial z}{\partial y} = 1$。加法节点是"梯度分流器"——上游梯度原封不动地复制给两个输入。
- **乘法** $z = x \cdot y$：$\dfrac{\partial z}{\partial x} = y,\ \dfrac{\partial z}{\partial y} = x$。乘法节点是"梯度交换器"——传给 $x$ 的梯度要乘以另一个输入 $y$ 的值，反之亦然。
- **tanh** $z = \tanh(x)$：$\dfrac{\partial z}{\partial x} = 1 - \tanh^2(x) = 1 - z^2$。方便的是，本地梯度可以直接用前向已经算好的输出 $z$ 表达，不必重算。

有了这些，反向传播的算法就是：

1. 沿计算图做**拓扑排序 (topological sort)**，得到一个所有节点的线性顺序，保证每个节点都排在它的输入之后。
2. 把输出节点（标量损失 $L$）的梯度置为 $1$——因为 $\partial L/\partial L = 1$。
3. 按拓扑序**逆序**遍历每个节点，调用它的本地反向规则，把梯度累加到它的输入上。

第 2 步的 $\partial L/\partial L = 1$ 是整台机器的点火钥匙：所有梯度都从这个 $1$ 出发，逆流而上，一路右乘本地梯度，流到每一个叶子（权重）。

### `+=` 不是笔误：一个必须亲眼看到的分叉

第 3 步为什么要**累加**梯度、而不是赋值？这是全章最容易错、也最该看清的一处，所以我们不空口断言，当场走一个最小的例子。

设 $b = -1$，令 $L = b\cdot b + b$。注意 $b$ 在这张图里**被用了两次**：一次进乘法（那条 $b\cdot b$ 的边），一次进加法（那条 $+\,b$ 的边）。这就是"分叉 (fan-out)"——一个节点分头喂给两条下游路径。

反向时，梯度顺着这两条路各自流回 $b$：

- 沿乘法那条路，$L$ 对 $b\cdot b$ 求导后再乘本地梯度，回到 $b$ 的贡献是 $2b = -2$；
- 沿加法那条路，回到 $b$ 的贡献是 $+1$。

多元链式法则的规矩是：一个变量若经由多条路径影响损失，各条路径的贡献要**相加**。所以

$$\frac{\partial L}{\partial b} = \underbrace{2b}_{\text{乘法路}} + \underbrace{1}_{\text{加法路}} = -2 + 1 = -1.$$

翻译成代码，就是每条路回传时对 `b.grad` 做 `+=`，让两份贡献叠加；若写成 `=`，后一条路会覆盖前一条，你只会拿到 $+1$，梯度就错了。稍后你会在引擎里看到，这个 $-1$ 是它自动跑出来的——`b.grad = -1.0`，两份贡献确实相加了。**每个 `_backward` 里写 `+=` 不是风格，是正确性**。记牢这张分叉图，它就是后面所有 `+=` 的理由。

### 为什么是反向模式，而不是前向模式

> **可跳过提示**：下面这段是全章抽象度最高的一节（雅可比连乘、结合律、代价对比）。如果你此刻更想动手，完全可以先直奔 ③ 把引擎造出来、把 XOR 训起来，回头再读这段，你会秒懂它在讲什么。它回答的是"我们造的这台机器，为什么方向选对了"。

你可能会问：为什么非要从输出端逆着来？能不能从输入端顺着推？能，那叫**前向模式 (forward mode)**。理解两者的分野，是理解整个深度学习为什么可行的关键。

回到 Ch9 的雅可比视角。一个网络是一串函数复合 $L = f_n \circ \cdots \circ f_2 \circ f_1$，总的导数是一串雅可比矩阵连乘 $J_n \cdots J_2 J_1$。矩阵乘法满足结合律，你可以任选结合顺序，结果一样，但**代价天差地别**。

关键事实是：损失 $L$ 是一个**标量**，而参数有几百万个。所以最左边的雅可比 $J_n$ 是一个 $1 \times m$ 的行向量，最右边关联的是海量输入。

- **反向模式**从左边（标量端）起手：先算 $J_n$（一个行向量），右乘 $J_{n-1}$，得到的还是一个**向量**；再右乘 $J_{n-2}$，还是向量……全程累积对象都是**向量**，每一步是"向量 × 矩阵"。跑完一遍，你一次性拿到损失对**所有**输入的梯度。总代价与一次前向传播**同阶**。
- **前向模式**从右边（输入端）起手：每次只能追踪损失对**一个**输入的导数，想拿全部 $m$ 个，就得把整个过程跑 $m$ 遍。对两百万参数，就是两百万倍的前向代价。

一句话：**多输入、单输出**的场景——这正是"几百万参数、一个标量损失"的神经网络——反向模式的优势是压倒性的，恰好等于参数量之比。这不是工程上的取巧，是 Ch9 那个"从标量端起手、按结合律从右往左乘最省"的结论在这里兑现。神经网络的损失地形永远是标量出口，于是反向模式永远是对的选择。

所以，反向传播 = 链式法则 + 一个聪明的计算顺序（拓扑序上先前向求值、再逆序回传梯度）。没有玄学。接下来我们把它变成能跑的东西。

## ③ 工程线（The How）——里程碑①：从零造一台 micro-autograd

现在造引擎。设计只有一个核心抽象：一个 `Value` 类，包住一个标量。它记住三样东西——自己的数值 `data`、自己的梯度 `grad`、以及**当初是怎么被算出来的**（指向父节点的引用，加一个闭包 `_backward`，封装"如何把上游梯度分发给我的输入"）。每做一次运算，我们不但算出数值，还顺手把这条反向规则挂上去。这样一张计算图就在前向传播的过程中被**自动、隐式地**建起来了，不需要你显式声明图结构。这个设计，正是 PyTorch 的 autograd、Karpathy 的 micrograd 的最小内核。

代码从上到下可直接运行，只依赖 `numpy` 和标准库，随机种子固定。整台 autograd 内核不过一百行上下。

```python
import numpy as np
import math

class Value:
    """一个包住标量的节点,记录数值、梯度,以及如何反向传播。"""
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None      # 默认叶子:什么都不做
        self._prev = set(_children)         # 指向父节点(输入)的引用
        self._op = _op                      # 记录是哪种运算,便于调试

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            # 上游梯度 out.grad × 本地梯度 1 → 分发给输入,这就是链式法则
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            # 上游梯度 out.grad × 本地梯度(另一输入的值) → 分发,这就是链式法则
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            # 上游梯度 out.grad × 本地梯度 n·x^(n-1) → 分发,这就是链式法则
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            # 上游梯度 out.grad × 本地梯度(1 - t^2,复用前向输出 t) → 分发,这就是链式法则
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    # 一些便利运算,全部由上面几个原子组合而成
    def __neg__(self):        return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other):return self + other
    def __rmul__(self, other):return self * other
    def __truediv__(self, o): return self * o ** -1

    def backward(self):
        # 1) 拓扑排序:保证每个节点排在它的输入之后
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        # 2) 清零 + 点火:输出对自己的导数是 1
        for v in topo:
            v.grad = 0.0
        self.grad = 1.0
        # 3) 逆拓扑序回传,每个节点把梯度分发给输入
        for v in reversed(topo):
            v._backward()
```

停下来看清楚三件事。第一，计算图是在你写 `a*b + c` 这样的普通表达式时**自动**长出来的——每个运算符重载都偷偷 new 了一个 `out` 节点，并把父节点塞进 `_prev`。你从没显式说"这是一张图"。第二，`backward()` 里的 `build` 是一个标准的深度优先拓扑排序：先递归处理所有输入，再把自己 append，于是 `topo` 里输入一定在前、输出在后，`reversed` 后就是从损失往回。第三，四个原子的 `_backward` 注释我特意写成同一句话的四次变奏——"上游梯度 × 本地梯度 → 分发给输入，这就是链式法则"——因为它们本就是同一条法则的四次套用，没有一处是新数学；而每处都用 `+=`，对应上一节那张分叉图的"多路径求和"。

一个诚实的边界要点：`backward()` 的清零只发生在**从损失可达**的那些节点上。本章的 MLP 里每个参数每一步都参与前向、必然可达，所以这样做完全正确、还顺手省了手动清零。但如果你日后改造代码，引入了某次前向没用到的参数（比如条件分支、跳层），它的 `.grad` 就不会被这次 `backward` 清零，陈旧的梯度会残留下来被误用。真到那一步，就得在优化器一侧显式清零，而不能再依赖"可达性清零"这个便利。

### 梯度检验：证明引擎没骗你

一台自动微分引擎最危险的失败模式，是它**看起来在工作、数值也不报错，但梯度是错的**。这种 bug 会让训练悄悄发散或卡住，你却找不到原因。所以每造一台 autograd，第一件事必须是**梯度检验 (gradient checking)**——这是 Ch9、Ch7 就立下的规矩。

方法回到导数的定义：$\dfrac{\partial L}{\partial x} \approx \dfrac{L(x+h) - L(x-h)}{2h}$，用一个极小的 $h$ 做**有限差分 (finite difference)**，得到一个纯数值的、不依赖任何链式法则的梯度。它慢（每个参数要跑两遍前向），但它是**独立的真值来源**。拿它对照 autograd 算出的解析梯度，若相对误差小到浮点噪声量级（<$10^{-6}$），就证明引擎对了。

要让这个证明有分量，检验表达式必须把引擎暴露的**每一个原子都走一遍**——加、乘、正整数幂、tanh，以及除法（它在内部走的是负幂 `**-1`，是最容易被漏测的一条路径）。所以下面的 $f$ 里特意塞了一个除法项：

```python
def grad_check():
    a = Value(-4.0); b = Value(2.0)
    d = a * b + b**3
    e = ((a + b).tanh() + d) * a
    f = e * e / (a + 6.0)       # 除法(负幂)也纳入检验,原子全覆盖
    f.backward()
    ag, bg = a.grad, b.grad

    h = 1e-6
    def forward(av, bv):        # 一个纯数值的前向,不含任何反向逻辑
        a = Value(av); b = Value(bv)
        d = a * b + b**3
        e = ((a + b).tanh() + d) * a
        return (e * e / (a + 6.0)).data
    num_a = (forward(-4.0+h, 2.0) - forward(-4.0-h, 2.0)) / (2*h)
    num_b = (forward(-4.0, 2.0+h) - forward(-4.0, 2.0-h)) / (2*h)
    print("grad a: auto=%.8f num=%.8f rel=%.2e"
          % (ag, num_a, abs(ag-num_a)/(abs(num_a)+1e-12)))
    print("grad b: auto=%.8f num=%.8f rel=%.2e"
          % (bg, num_b, abs(bg-num_b)/(abs(num_b)+1e-12)))
```

运行后你会看到（数值可核验）：

```
grad a: auto=-39.37342546 num=-39.37342547 rel=1.38e-10
grad b: auto=-124.48527975 num=-124.48527976 rel=7.62e-11
```

相对误差 $\sim 10^{-10}$，远小于 $10^{-6}$。解析梯度和有限差分几乎完全一致——引擎的每个原子（含那条除法/负幂路径）都被独立核验，全对。这个断言不是仪式，它是你敢用这台引擎去训练网络的全部底气。

### 终于：训练 Ch22 那个网络

现在兑现承诺。用这台引擎搭一个 MLP——每个神经元是"权重点乘输入、加偏置、过 tanh"，一层是若干神经元，一个 MLP 是若干层堆叠。注意：整个网络里没有一行反向传播的代码，我们只写**前向**；梯度会由 `Value` 在幕后自动接好。

```python
class Neuron:
    def __init__(self, nin, rng):
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)  # w·x + b
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
        self.layers = [Layer(sz[i], sz[i+1], rng) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def train_xor():
    rng = np.random.RandomState(1)          # 固定种子,结果可复现
    model = MLP(2, [4, 4, 1], rng)          # 2 输入 → 4 → 4 → 1 输出
    X = [[0.,0.], [0.,1.], [1.,0.], [1.,1.]]
    Y = [-1., 1., 1., -1.]                   # XOR,用 ±1 配合 tanh 输出
    for epoch in range(200):
        ypred = [model(x) for x in X]
        loss  = sum(((yp - yt)**2 for yp, yt in zip(ypred, Y)), Value(0.0))
        loss.backward()                      # 一句话,全部梯度自动算出
        for p in model.parameters():
            p.data -= 0.1 * p.grad           # 梯度下降(Ch14),沿负梯度走一步
        if epoch % 20 == 0 or epoch == 199:
            print("epoch %3d loss %.6f" % (epoch, loss.data))
    preds = [model(x).data for x in X]
    acc = sum((p>0)==(t>0) for p,t in zip(preds, Y)) / len(Y)
    print("final preds:", ["%.3f" % p for p in preds])
    print("accuracy:", acc)

grad_check(); print("---"); train_xor()
```

`loss.backward()` 这一行，是整章的高潮：它触发拓扑排序、点火 $\partial L/\partial L=1$、逆序回传，把损失对网络里**每一个**权重和偏置的梯度算好，填进各自的 `.grad`。然后 Ch14 的梯度下降接手，沿负梯度更新一步。前向、反向、更新——这三步就是深度学习的全部训练循环。

这里要澄清一个常被含糊过去的细节，免得你日后上手 PyTorch 时踩坑。我们的 `backward()` 在每次调用开头就把可达节点的 `.grad` 清零了，所以循环里不必显式清梯度。**PyTorch 不这样**：它的 `.grad` 默认是**跨调用累加**的（这是为了支持手动跨小批次累加梯度），因此真实训练循环是**三步**——`optimizer.zero_grad(); loss.backward(); optimizer.step()`，忘了第一步是新手最著名的坑之一。换句话说，我们把"清零"内建进了 `backward`，PyTorch 把它单独交给了你。精神一致，接口有别。注意这跟本章反复强调的 `+=` 并不矛盾：`+=` 累加的是**同一次** `backward` 内、同一节点的多条路径贡献（那张分叉图）；清零发生在**两次** `backward` 之间。

运行后你会看到（数值可核验）：

```
epoch   0 loss 4.670555
epoch  20 loss 2.816518
epoch  40 loss 0.261855
epoch  60 loss 0.083966
epoch  80 loss 0.047099
epoch 100 loss 0.032019
epoch 120 loss 0.023999
epoch 140 loss 0.019076
epoch 160 loss 0.015769
epoch 180 loss 0.013404
epoch 199 loss 0.011712
final preds: ['-0.940', '0.958', '0.949', '-0.939']
accuracy: 1.0
```

损失从 4.67 一路跌到 0.012。四个预测——$(-0.94, 0.96, 0.95, -0.94)$——符号完全对上 XOR 的 $(-1, +1, +1, -1)$，**准确率 1.0**。

停下来体会这一刻。Ch1 我们说感知机学不了 XOR，是它把整个连接主义打入冷宫二十年的那个反例。Ch22 我们用 UAT 论证一个隐藏层的 MLP **能**表示 XOR，却只搭好没训练——因为我们还没有算梯度的机器。现在，这台你亲手从一百多行代码造出来的 micro-autograd，把那个搭好却训不了的网络**教会了**。那份"搭好却训不了"的悬而未决，到此被彻底了结：找权重，靠的就是这台机器。

## ④ 研究线（The Where）：一个想法点燃一个时代

你刚造的东西，不是玩具的另一种说法，而是**现代深度学习框架的心脏**。把 PyTorch、JAX、TensorFlow 剥到最里层，你会发现它们本质上是同一个方程式：

$$\text{框架} = \text{autograd} + \text{硬件加速 (GPU/TPU)} + \text{神经网络算子库}$$

你造的 `Value` 是标量版的 autograd；把它换成包住多维数组的张量、把每个 `_backward` 换成 GPU 上的并行核函数、再配上卷积/注意力等成百上千个预写好的算子——你就得到了 PyTorch。核心那台"自动跑链式法则"的引擎，和你写的一模一样。这条路线直接致敬 Karpathy 的 micrograd：一百行标量 autograd，却把 PyTorch 的灵魂讲透了——你这一章写的引擎，规模与它同量级。

前向模式并没有消失——它在"少输入、多输出"的场景（如某些科学计算、雅可比向量积）里反而更优，JAX 同时提供 `jax.grad`（反向）和 `jax.jvp`（前向）。取舍永远回到那句：看你要的是 $\partial(\text{标量})/\partial(\text{海量})$ 还是反过来。神经网络永远是前者，所以反向模式统治了这个领域。

值得记住一段历史。反向传播的核心思想在 1960–70 年代就零散出现过（Linnainmaa 1970 的自动微分、Werbos 的博士论文），但真正让它进入所有人视野的，是 1986 年 Rumelhart、Hinton、Williams 那篇《Learning representations by back-propagating errors》。他们没有发明新数学——链式法则是牛顿-莱布尼茨时代就有的——他们做的，是指出**把链式法则机械化地铺在多层网络上，就能自动学出隐藏层的表示**。正是这一个想法，让训练任意深、任意宽的网络在原理上成为可能，几十年后配上 GPU 和大数据，点燃了整个深度学习时代。你今天用一百多行代码复现的，就是那个点火装置。

**一个可动手验证的问题**：把 `train_xor` 里的 MLP 从 `[4,4,1]` 加深到 `[4,4,4,4,4,4,1]`（六层隐藏层），其余不变，损失还降得下去吗？

先说诚实的答案：**看运气**。我实测过，`RandomState(1)` 和 `RandomState(3)` 时这个深网卡在 loss $\approx 2.7$、$3.9$ 训不动；但 `RandomState(7)`、`RandomState(11)` 却能收敛到 loss $\approx 0.002$。加深**不是**必然失败——初始化的运气占了很大分量，这本身就值得记一笔。但无论收敛与否，有一个现象稳定出现：**打印每一层参数 `.grad` 绝对值的均值，你会看到靠近输入的那几层，梯度普遍比靠近输出的那几层小上一到两个数量级**（比如 seed 1 时，输入侧各层约 $10^{-3}$，输出侧约 $8\times10^{-2}$）。深度越深，信号从损失端一路右乘本地梯度回传时衰减得越厉害，最前面的层几乎收不到有效的更新信号。

这不是引擎坏了——梯度检验依然会通过，算出来的每个梯度都精确正确；这是训练本身的病理，它有个名字叫**梯度消失 (vanishing gradient)**（tanh 的饱和会加剧它）。原来，**能算梯度，离能训练好一个网络，还差着一整门学问**。这门学问，正是下一章的入口。

## ⑤ 检索式自测

先自己答，再展开核对。

1. UAT 保证了什么、没保证什么？这个"没保证"为什么把我们逼向反向传播？
2. 用一句话说清反向传播的核心动作（上游、本地、分发三个词都要出现）。
3. 加法节点、乘法节点在反向时分别对梯度做了什么（给它们各起一个绰号）？
4. 为什么损失是标量这件事，决定了反向模式而非前向模式统治深度学习？
5. 为什么每个 `_backward` 里必须用 `+=` 而不是 `=`？举一个会出错的场景。
6. 梯度检验用什么独立方法产生"真值"？通过的判据是什么、说明了什么？

<details>
<summary>展开答案</summary>

1. UAT 保证**存在**一组权重能把目标函数逼近到任意精度（存在性），但不告诉你这组权重**在哪**、怎么到达。学习的实质是"找到"它们；找权重靠梯度下降，梯度下降要梯度，而几百万个分量的梯度手推不可能——只能靠反向传播自动算。
2. 每个节点把**上游**传来的梯度乘以自己的**本地**梯度（该运算的偏导），再**分发**给它的输入。这就是链式法则。
3. 加法是"梯度分流器"：本地梯度都是 1，把上游梯度原样复制给两个输入。乘法是"梯度交换器"：传给 $x$ 的梯度乘以另一个输入 $y$ 的值，传给 $y$ 的乘以 $x$。
4. 一串雅可比连乘可任选结合顺序，结果同、代价异。损失是标量意味着最左的雅可比是行向量，从标量端起手右乘，累积对象全程是**向量**，一遍反向就拿到对所有参数的梯度，代价与前向同阶；前向模式每次只追踪对一个输入的导数，要跑参数量那么多遍。多输入单输出正是神经网络的形状，故反向模式胜。
5. 因为一个节点可能被多个下游用到，多元链式法则要求各路径贡献**相加**。用 `=` 会让后一条路径覆盖前一条，梯度算错。场景：正文那个 $L=b\cdot b+b$，$b$ 分叉给乘法和加法两条路，`+=` 才能把 $2b$ 与 $1$ 相加得 $-1$；换成 `=` 只剩 $+1$。同理"同一个权重出现在多处"也是分叉。
6. 用**有限差分**（中心差分 $\frac{L(x+h)-L(x-h)}{2h}$）独立产生数值梯度，不依赖任何链式法则。判据是解析梯度与数值梯度的相对误差小到浮点噪声量级（<$10^{-6}$，本章实测 $\sim10^{-10}$）。通过说明 autograd 引擎的反向规则实现正确。

</details>

## ⑥ 白板自检清单

拿一支笔，不看书，完成以下每一项：

- 画一张 $L=(\tanh(w x + b)-y)^2$ 的计算图，标出每个节点；沿图跑一遍前向，再逆序跑一遍反向，写出每条边上传递的梯度。
- 画一张有 fan-out 的小图（如 $L=b\cdot b+b$），亲手让梯度从两条路回到 $b$、相加，验证 `+=` 的必要性。
- 默写加、乘、tanh 三个运算的本地梯度，并用同一句"上游×本地→分发"说明每个 `_backward` 为什么长成那样。
- 只凭记忆写出 `backward()` 的三步：拓扑排序、点火 $\partial L/\partial L=1$、逆序回传；说清为什么拓扑序不可省。
- 解释"反向传播 = 链式法则 + 聪明的计算顺序"，并说清那个顺序聪明在哪。
- 说清梯度检验的原理与判据，以及不做它的风险。
- 讲清 PyTorch 和你的 micro-autograd 在核心内核上有何相同、有何不同（提示：梯度清零。）

## ⑦ 回到中心对象，与下一章的钩子

现在把那句被我们悬置整章的话，完整取出来：**智能是在不确定性与资源约束下从经验中获得达成目标的能力；学习是用数据做归纳推断**。整本书我们都在逼近它，而本章第一次让其中"**从经验中**获得能力"这半句，从口号变成了一段能真正运行的代码。

Ch22 我们有了能表示复杂函数的网络（容量），Ch14 我们有了改进权重的机制（梯度下降），但两者之间缺一座桥——**怎么把网络里两百万个权重各自该往哪挪算出来**。本章造的这台自动微分引擎，就是那座桥。它让训练成为一个闭合的循环：喂进数据，前向算损失，`backward()` 让梯度沿计算图逆流而上填满每一个参数，梯度下降沿负梯度挪一步——网络于是**真的在从经验中改进**。我们第一次把 Ch14 的优化，严丝合缝地接到了 Ch22 的网络上。这是全书工程主脊的第一个里程碑：能算梯度了，能训练了。

但你在研究线末尾那个可动手验证的问题里，已经嗅到了危险的气味。把网络加深到六层，能不能训好开始**看初始化的运气**；而无论如何，最靠近输入那几层的梯度都会小上一两个数量级，权重几乎不动。这不是引擎坏了，梯度检验依然会通过；这是训练本身的病理。原来，**能算梯度，离能训练好一个网络，还差着一整门学问**。初始化不当，梯度会消失或爆炸；学习率错一个数量级，损失就发散；网络越深越难驯服。让训练真正 work，是一门科学，也是一门手艺。

下一章，Ch24《训练的科学与艺术》，我们就去驯服它。

---

**习题**

**数学题**

1.（★）推导减法 $z=x-y$、除法 $z=x/y$ 的本地梯度，并说明为什么本章代码能用 `__neg__` 和 `__pow__(-1)` 组合出它们而不必单独实现。
2.（★★）设 $z = \sigma(x)=\frac{1}{1+e^{-x}}$，证明 $\frac{dz}{dx}=z(1-z)$；再对 $z=\text{ReLU}(x)$ 写出本地梯度，并指出它在 $x=0$ 处的处理为什么在实践中无关紧要。
3.（★★）推导矩阵乘法 $Z = XW$ 的反向规则：给定上游梯度 $\frac{\partial L}{\partial Z}$，证明 $\frac{\partial L}{\partial X}=\frac{\partial L}{\partial Z}W^\top$、$\frac{\partial L}{\partial W}=X^\top\frac{\partial L}{\partial Z}$。这是把标量 autograd 升级到张量 autograd 的关键一步。
4.（★★★）证明：对一个由 $E$ 条边组成的计算图、单标量输出，反向模式一次算出全部输入梯度的运算量是 $O(E)$，与一次前向同阶；而前向模式要 $O(nE)$，$n$ 为输入数。
5.（★★）在纸上取小图 $L=(a\cdot b + b)^2$，$a=2,b=-1$，手工跑一遍前向和反向，写出 $\partial L/\partial a$、$\partial L/\partial b$，并用中心差分验算。（注意 $b$ 分叉两处，`+=` 在此必不可少。）

**编程题**

6.（★）给 `Value` 加上 `exp`、`relu`、`log` 三个运算（各写好前向与 `_backward`），并把它们纳入梯度检验。
7.（★★）把 `grad_check` 扩展成**全网梯度检验**：对训练好的 MLP，随机扰动每个参数做有限差分，断言与 autograd 梯度的相对误差 <$10^{-5}$。
8.（★★）观察梯度消失：把 MLP 加深到 `[4,4,4,4,4,4,1]`，每个 epoch 打印各层参数 `.grad` 绝对值的均值。**先对比输入端一层与输出端一层的量级差**（你会看到稳定的一到两个数量级差距），再看中间层——你会发现它并非单调衰减，而是带噪声的；请解释为什么"端点对比"比"整条曲线单调下降"是更稳的观察量。顺便换几个随机种子（如 1、3、7、11），记录哪些能收敛、哪些卡住，体会初始化的影响。
9.（★★★）把标量 `Value` 重写为包住 NumPy 数组的 `Tensor`，实现第 3 题的 matmul 反向规则，用它把 XOR 的训练从"逐样本"改成"整批矩阵一次前向"，对比速度。

**思考题**

10.（★★，开放）为什么是反向模式、而不是前向模式，统治了深度学习？如果未来出现"少参数、多输出"的主流模型，这个结论会翻转吗？
11.（★★，开放）反向传播需要在前向时把所有中间结果都存下来（才能在反向时用）。这带来的内存代价随网络深度如何增长？"梯度检查点 (gradient checkpointing)"用时间换内存的思路大概是什么？

---

**延伸阅读**

- Karpathy，《Neural Networks: Zero to Hero》与 micrograd 代码库——本章 micro-autograd 的直接精神来源，一百行讲透 autograd，配套视频手把手推每一行，读完你会觉得 PyTorch 不再是黑箱。
- Rumelhart, Hinton & Williams (1986)，《Learning representations by back-propagating errors》——把反向传播推向世界的那篇，读它是为了看清"机械化链式法则"这个想法在历史现场的分量。
- Goodfellow, Bengio & Courville，《Deep Learning》第 6.5 节——对反向传播作为计算图上一般算法的严谨、系统的表述，补齐本章为清晰而略去的形式细节。
- Baydin, Pearlmutter, Radul & Siskind (2018)，《Automatic Differentiation in Machine Learning: a Survey》——把前向/反向模式、AD 与符号/数值微分的关系一次讲清的权威综述，读它建立完整地图。
- PyTorch autograd 官方文档与 `torch.autograd.gradcheck`——看工业级 autograd 如何组织、`.grad` 为何默认累加需要 `zero_grad`，以及框架自带的梯度检验工具长什么样，把你的玩具引擎接回真实世界。