# 第 26 章 · 卷积网络与视觉的归纳偏置

## ① 概念主线：把先验烤进架构

上一章我们看到，一个多层感知机 (multilayer perceptron, MLP) 原则上可以从数据里学出某种层级化的表示——但不保证、也不受任何空间结构约束。这是一个解放性的结论：不必再手工设计特征，梯度会替你把它们从数据里长出来。但那一章的结尾我埋了一根刺，现在要把它拔出来看清楚。

请你做一件具体的事。取一张 $100\times100$ 的灰度图,把它拉平成一个 $10000$ 维的向量,喂给一个只有 $1000$ 个隐藏单元的全连接层。这一层的权重矩阵有多大?$10000\times1000=10^7$,一千万个参数。而这仅仅是第一层,仅仅是一张小图。你还没开始学任何东西,参数量就已经失控了。这是第一宗罪:**参数爆炸**。

第二宗罪更隐蔽,也更致命。设想训练集里有一张猫的照片,猫在画面正中央;MLP 辛辛苦苦学会了"中央这一片像素呈现这种模式时,是猫"。现在来了一张几乎相同的照片,只是猫挪到了左上角。对你我而言,这显然还是同一只猫。但对 MLP 呢?左上角的像素对应的是权重矩阵里**完全不同的一批行**。中央学到的那套权重,对左上角一无所知。网络必须把"猫"这件事在每一个可能的位置**从头再学一遍**。图像里"平移一下还是同一个东西"这条无比基本的事实,MLP 天生不知道——它把 $10000$ 个像素看成 $10000$ 个彼此无关的独立坐标,像对待一张打乱了顺序的电子表格。这是第二宗罪:**无视空间结构、且不平移不变**。

回到 Ch5 的语言:选一个表示,就是下一次注。MLP 下的注是"我对输入的结构一无所知,任何两个维度都可能任意相关"。这个注在图像上是个坏注,因为图像的结构我们**明明知道**:相邻像素高度相关(局部性),而一个模式平移之后还是同一个模式(平移不变性)。既然我们知道,为什么要让网络花掉海量数据、海量参数,去重新发现这两条我们早就写在纸上的先验?

这一章的核心思想只有一句话:**与其让网络从数据里摸索出图像的结构先验,不如把这些先验直接写进架构本身**。做法有两条,合起来就是卷积:

- **局部感受野 (local receptive field)**:每个神经元只看输入里一小块邻近的像素,而不是全部。这直接编码了"局部性"。
- **权重共享 (weight sharing)**:同一组权重(一个滤波器)在图像的所有位置**复用**。中央学到的"猫腿检测器",挪到左上角照样能用。这直接编码了"平移不变"。

这就是本章的纲领:**架构即归纳偏置 (architecture as inductive bias)**。不是把先验塞进损失函数、也不是塞进数据增强,而是塞进网络的连接方式。卷积网络之所以是深度学习设计思想的第一个、也是最干净的范例,正因为它把"我们知道图像长什么样"这件事,变成了钢筋水泥般的结构约束。下面我们先把数学讲透,再从零把它实现出来、训练出来。

---

## ② 数学线(The Why)

### 卷积操作

我们沿用 Ch4 的张量约定:一批图像是形状 $(N, C, H, W)$ 的四维张量——$N$ 是 batch,$C$ 是通道 (channel),$H, W$ 是高和宽。

一个**滤波器 (kernel/filter)** 是一个小张量,记 $\mathbf{K}\in\mathbb{R}^{C\times k_h\times k_w}$,通常 $k_h,k_w$ 只有 $3$ 或 $5$。它在输入上滑动,每滑到一个位置,就与覆盖下的那一小块做**局部加权和**,产出一个数。所有位置的输出拼起来,就是一张**特征图 (feature map)**。

写成公式(为清晰只写单通道输入、单滤波器,位置 $(i,j)$):

$$
S(i,j) \;=\; \sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1} X(i+u,\,j+v)\,\cdot\,K(u,v)\;+\;b.
$$

严格说这是**互相关 (cross-correlation)**;数学上的卷积会把核翻转 $K(-u,-v)$。深度学习里几乎一律用互相关,因为核是学出来的,翻不翻转只是重参数化,不影响表达能力。我们全程沿用互相关,但仍按惯例叫它"卷积"。

一层通常有 $C_{\text{out}}$ 个滤波器,每个吃进全部 $C_{\text{in}}$ 个输入通道,于是权重张量形状是 $(C_{\text{out}}, C_{\text{in}}, k_h, k_w)$,输出是 $(N, C_{\text{out}}, H_o, W_o)$。在最常见的 "valid" (无填充、步长 1) 设定下,输出尺寸为

$$
H_o = H - k_h + 1,\qquad W_o = W - k_w + 1.
$$

记住这个形状公式——它是本章第一道习题,也是你调试代码时第一个要在脑子里算对的东西。

### 权重共享 → 平移等变

现在把第二宗罪的解药讲成一个可证明的性质。设 $T_\delta$ 表示"把图像沿宽度方向平移 $\delta$ 列"的算子:$(T_\delta X)(i,j) = X(i, j-\delta)$。卷积算子记作 $\text{Conv}_K$。**平移等变 (translation equivariance)** 断言:

$$
\text{Conv}_K(T_\delta X) \;=\; T_\delta\big(\text{Conv}_K(X)\big).
$$

也就是说,**先平移再卷积,等于先卷积再平移**。输入挪了一格,特征图原封不动地也跟着挪一格。

证明是一行代数。设 $S = \text{Conv}_K(X)$,考察平移后的输入在位置 $(i,j)$ 的卷积输出:

$$
\big[\text{Conv}_K(T_\delta X)\big](i,j) = \sum_{u,v}(T_\delta X)(i+u,j+v)\,K(u,v) = \sum_{u,v}X(i+u,\,j+v-\delta)\,K(u,v) = S(i,\,j-\delta) = \big[T_\delta S\big](i,j).
$$

关键的一步是:核 $K(u,v)$ **不依赖于位置 $(i,j)$**——正是权重共享。如果每个位置有自己的一套权重(那就退化成全连接了),这个求和里就没法把 $\delta$ 干净地提出来,等变性立刻破碎。所以:**权重共享是平移等变的充分来源**——它一定给你等变。逆向也成立,但要加个前提:在**线性算子**范围内,平移等变的算子必然是卷积(即必然权重共享),这是一条定理;跳出线性,等变未必逼出权重共享。(边界处会有小小的例外:平移会把内容移出画面、或从另一边卷入,所以等变严格成立只在"有效区域"内——工程线里我们会把这条容差核验成一个具体数字。)

### 等变 ≠ 不变,池化补上不变

请分清两个词。**等变 (equivariance)**:输入变换,输出以**对应方式**跟着变换(平移→平移)。**不变 (invariance)**:输入变换,输出**纹丝不动**。卷积给的是等变,不是不变——猫挪了,特征图也挪了,而不是特征图不变。

但很多任务(比如"图里有没有猫")要的是不变:不管猫在哪,答案都是"有"。怎么从等变得到不变?**池化 (pooling)**。最大池化 (max pooling) 把特征图上每个 $2\times2$ 小窗里取最大值,输出缩小一半。直觉是:"这一小片区域里有没有强响应"这个信息,对模式在小片内部具体落在哪个像素并不敏感。局部池化带来**局部平移不变性**,同时顺手做了**下采样**(减小尺寸、扩大后续层的感受野)。堆叠"卷积(等变)+ 池化(局部不变)"若干次,不变性就一层层累积,最终读出层看到的表示对小到中等幅度的平移近似不变。诚实地说:实际 CNN 因步长下采样带来的混叠 (aliasing),并非严格平移不变——"如何让 CNN 真正平移不变"(Zhang 2019)至今是一个活跃的研究点。

### 参数效率

现在把第一宗罪的解药量化。一个把 $H\times W\times C_{\text{in}}$ 输入全连接到 $H_o\times W_o\times C_{\text{out}}$ 输出的全连接层,参数量是

$$
\underbrace{(H\,W\,C_{\text{in}})}_{\text{输入}}\times\underbrace{(H_o W_o C_{\text{out}})}_{\text{输出}}\;\;(+\,H_o W_o C_{\text{out}}\ \text{个偏置}),
$$

一个巨大的乘积。而卷积层的参数量与图像大小**完全无关**:

$$
C_{\text{out}}\times C_{\text{in}}\times k_h\times k_w\;+\;C_{\text{out}}.
$$

前者是 $O(\text{输入}\times\text{输出})$,后者是 $O(\text{核}\times\text{通道})$。局部性砍掉了"每个输出连所有输入",权重共享砍掉了"每个位置一套参数"。工程线里你会看到,同一个任务上这个比值是**上千倍**。

这不只是省内存。回到 Ch11 与 Ch15:参数少意味着**有效容量低**,而合适的先验(局部+平移)恰好匹配了图像的真实结构,于是我们在偏差-方差权衡里两头得利——既不像全连接那样容易过拟合,又因为先验对路而不引入额外偏差。这正是 Ch15 "no free lunch" 的正面兑现:**当先验与数据结构匹配时,收益是实打实的正——代价并没有消失,而是被转嫁到了它不匹配的那些分布上**(后文"诚实边界"会让你亲眼看到代价现身)。

### 层级特征

单层卷积的感受野只有 $k\times k$。但把卷积层堆起来——每一层的一个神经元看下一层的一小块,而下一层的每个点又汇聚了再下一层的一小块——**感受野随深度扩大**:纯堆叠 stride-1 卷积时它按 $\text{RF}=1+\sum_l(k_l-1)$ **线性**增长,而一旦配上池化 / 步长下采样,感受野便成倍(近似指数级)扩张。于是自然涌现出层级:低层滤波器学到边缘、颜色梯度、纹理;中层把边缘组合成角点、条纹、简单形状;高层把部件拼成物体。这正是 Ch25 表示学习那条"边→部件→物体"的层级组合,只不过现在它被架构**强制**成了空间上的层级,而不是碰运气学出来的。

### 诚实边界

卷积的先验只在**数据真的具有局部性和平移结构时**才是福。想象你把每张图像的像素按一个固定的随机置换彻底打乱——信息一点没丢(置换可逆),一个全连接网络的表现完全不受影响,因为它本来就不在乎顺序。但卷积网络会**崩溃**:相邻像素不再相关,"局部窗口"里全是无关的东西,平移等变对应的也不再是任何有意义的变换。对表格数据(每一列是收入、年龄、邮编,列之间没有空间邻接、平移毫无意义)也一样:卷积的偏置是**错的偏置**。这是 Ch15 的另一面——先验是一把双刃剑,匹配则升,错配则降。工程线的最后一道习题会让你亲手把像素打乱,看着 CNN 的优势蒸发,把这条边界变成你亲眼见过的事实。

---

## ③ 工程线(The How)——里程碑②:从零训练一个小 CNN

现在兑现里程碑。下面是一个单文件、纯 NumPy、固定随机种子、从上到下可直接运行的完整脚本。它:(1) 从零实现卷积的前向与反向、$2\times2$ 最大池化的前向与反向;(2) 搭一个 "卷积 → ReLU → 池化 → 线性读出" 的小 CNN;(3) 在一个合成任务上从零训练——**区分横条与竖条,而条出现在图像的随机位置**(这个"随机位置"正是为了逼出平移不变的价值:MLP 得为每个位置重学,CNN 靠权重共享一次学会);(4) 数值验证平移等变;(5) 打印卷积 vs 全连接的参数量对比。

反向传播全部手写,直接兑现 Ch23:卷积的反向,本质是把上游梯度按同样的滑窗规则**散射**回权重和输入——$\mathrm{d}W$ 是"上游梯度 $\times$ 对应输入 patch"在所有位置上的累加,$\mathrm{d}x$ 是"上游梯度 $\times$ 核"散射回各 patch 位置的累加。你会看到,权重共享在前向是"一套核用在所有位置",在反向就自动变成"所有位置的梯度累加到同一套核"——这也是为什么卷积层参数少却能收到来自整张图的学习信号。

```python
import numpy as np

rng = np.random.default_rng(0)

# ---------- 合成数据：横条 vs 竖条，出现在随机位置 ----------
IMG = 12
BAR = 4  # 条的长度

def make_sample(kind):
    x = np.zeros((IMG, IMG), dtype=np.float64)
    if kind == 0:  # 横条
        r = rng.integers(0, IMG)
        c = rng.integers(0, IMG - BAR + 1)
        x[r, c:c+BAR] = 1.0
    else:          # 竖条
        r = rng.integers(0, IMG - BAR + 1)
        c = rng.integers(0, IMG)
        x[r:r+BAR, c] = 1.0
    x += 0.05 * rng.standard_normal((IMG, IMG))  # 一点噪声
    return x

def make_dataset(n):
    X = np.zeros((n, 1, IMG, IMG)); y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        k = rng.integers(0, 2)
        X[i, 0] = make_sample(k); y[i] = k
    return X, y

Xtr, ytr = make_dataset(400)
Xte, yte = make_dataset(200)

# ---------- 卷积（互相关），valid，多通道 ----------
def conv_forward(x, W, b):
    # x:(N,Cin,H,W)  W:(Cout,Cin,kh,kw)  b:(Cout,)
    N, Cin, H, Wd = x.shape
    Cout, _, kh, kw = W.shape
    Ho, Wo = H - kh + 1, Wd - kw + 1
    out = np.zeros((N, Cout, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            patch = x[:, :, i:i+kh, j:j+kw]                    # (N,Cin,kh,kw)
            out[:, :, i, j] = np.tensordot(patch, W, axes=([1,2,3],[1,2,3])) + b
    return out

def conv_backward(dout, x, W):
    N, Cin, H, Wd = x.shape
    Cout, _, kh, kw = W.shape
    Ho, Wo = dout.shape[2], dout.shape[3]
    dx = np.zeros_like(x); dW = np.zeros_like(W); db = dout.sum(axis=(0,2,3))
    for i in range(Ho):
        for j in range(Wo):
            patch = x[:, :, i:i+kh, j:j+kw]                    # (N,Cin,kh,kw)
            g = dout[:, :, i, j]                               # (N,Cout)
            dW += np.tensordot(g, patch, axes=([0],[0]))       # 权重共享→所有位置梯度累加
            dx[:, :, i:i+kh, j:j+kw] += np.tensordot(g, W, axes=([1],[0]))
    return dx, dW, db

# ---------- 2x2 最大池化 ----------
def maxpool_forward(x):
    N, C, H, Wd = x.shape
    Ho, Wo = H//2, Wd//2
    out = np.zeros((N, C, Ho, Wo)); mask = np.zeros_like(x)
    for i in range(Ho):
        for j in range(Wo):
            region = x[:, :, 2*i:2*i+2, 2*j:2*j+2].reshape(N, C, 4)
            idx = region.argmax(axis=2)
            out[:, :, i, j] = region.max(axis=2)
            for a in range(2):
                for bb in range(2):
                    mask[:, :, 2*i+a, 2*j+bb] = (idx == (a*2+bb))
    return out, mask

def maxpool_backward(dout, mask):
    N, C, Ho, Wo = dout.shape
    dx = np.zeros_like(mask)
    for i in range(Ho):
        for j in range(Wo):
            for a in range(2):
                for bb in range(2):
                    dx[:, :, 2*i+a, 2*j+bb] = mask[:, :, 2*i+a, 2*j+bb] * dout[:, :, i, j]
    return dx

def relu(x): return np.maximum(0, x)

def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(p[np.arange(N), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(N), y] -= 1; d /= N
    return loss, d

# ---------- 初始化：conv 12->10, pool 10->5, flatten 4*5*5=100 -> 2 类 ----------
Cout, k = 4, 3
W1 = rng.standard_normal((Cout, 1, k, k)) * 0.3; b1 = np.zeros(Cout)
FLAT = Cout * 5 * 5
W2 = rng.standard_normal((FLAT, 2)) * 0.1;       b2 = np.zeros(2)

def forward(x):
    c = conv_forward(x, W1, b1)
    a = relu(c)
    p, mask = maxpool_forward(a)
    flat = p.reshape(x.shape[0], -1)
    logits = flat @ W2 + b2
    return logits, (x, c, a, p, mask, flat)

def accuracy(X, y):
    logits, _ = forward(X)
    return (logits.argmax(1) == y).mean()

lr = 0.05
for epoch in range(30):
    perm = rng.permutation(len(Xtr))
    for s in range(0, len(Xtr), 32):
        idx = perm[s:s+32]; xb, yb = Xtr[idx], ytr[idx]
        logits, cache = forward(xb)
        loss, dlog = softmax_ce(logits, yb)
        x, c, a, p, mask, flat = cache
        dW2 = flat.T @ dlog; db2 = dlog.sum(0)
        dp = (dlog @ W2.T).reshape(p.shape)
        da = maxpool_backward(dp, mask)
        dc = da * (c > 0)                          # ReLU 反向
        _, dW1, db1 = conv_backward(dc, x, W1)
        W2 -= lr*dW2; b2 -= lr*db2; W1 -= lr*dW1; b1 -= lr*db1
    if epoch % 5 == 0 or epoch == 29:
        print(f"epoch {epoch:2d}  loss {loss:.4f}  "
              f"train_acc {accuracy(Xtr,ytr):.3f}  test_acc {accuracy(Xte,yte):.3f}")

print("FINAL train_acc", round(accuracy(Xtr,ytr),3), "test_acc", round(accuracy(Xte,yte),3))

# ---------- 平移等变数值验证 ----------
xs = Xte[:1].copy()
c0 = conv_forward(xs, W1, b1)
xs_shift = np.roll(xs, shift=1, axis=3)            # 输入向右平移一列
c1 = conv_forward(xs_shift, W1, b1)
c0_shift = np.roll(c0, shift=1, axis=3)            # 把原特征图也平移一列
diff = np.abs(c1[:, :, :, 1:] - c0_shift[:, :, :, 1:]).max()  # 忽略环绕的边界列
print("equivariance max diff (valid region):", round(float(diff), 8))

# ---------- 参数量对比 ----------
conv_params = W1.size + b1.size
fc_hidden = Cout*10*10                             # 等价全连接隐藏单元数
fc_params = 144*fc_hidden + fc_hidden
print("conv layer params:", conv_params, " equivalent FC params:", fc_params,
      " ratio:", round(fc_params/conv_params, 1))
```

**运行后你应当看到(固定种子 `default_rng(0)`,数值可逐条核验):**

```
epoch  0  loss 0.7154  train_acc 0.510  test_acc 0.520
epoch  5  loss 0.6775  train_acc 0.645  test_acc 0.660
epoch 10  loss 0.5491  train_acc 0.735  test_acc 0.730
epoch 15  loss 0.5523  train_acc 0.792  test_acc 0.780
epoch 20  loss 0.4771  train_acc 0.868  test_acc 0.910
epoch 25  loss 0.2977  train_acc 0.940  test_acc 0.965
epoch 29  loss 0.2998  train_acc 0.958  test_acc 0.970
FINAL train_acc 0.958 test_acc 0.97
equivariance max diff (valid region): 0.0
conv layer params: 40  equivalent FC params: 58000  ratio: 1450.0
```

三条可断言的性质,逐条对上了:

1. **从零训练到高准确率**。准确率从随机的 $0.51$(二分类基线 $0.5$)一路爬到测试集 $0.97$,远远优于瞎猜。而条出现在**随机位置**——CNN 靠权重共享的那 $9$ 个核参数,一次就学会了"横向 vs 纵向"这件与位置无关的事,不必为每个位置重学。这就是里程碑②的兑现:一个手写反向传播、纯 NumPy 的 CNN,真的从随机初始化学会了看。

2. **平移等变是可核验的事实,不是口号**。把输入向右平移一列、把原特征图也向右平移一列,在有效区域内两者的最大逐元素差是 **$0.0$**(数值上精确为零,因为卷积是确定性的、核处处相同)。$\S2$ 那一行代数,在这里变成了一个你能打印出来的 $0$。

3. **卷积 $\ll$ 全连接的参数量**。卷积层 $40$ 个参数,对应功能的全连接层 $58000$ 个,比值 **$1450$ 倍**。局部性与权重共享把参数砍掉了三个数量级——而它学得反而更好、泛化更稳(测试 $0.97$ 与训练 $0.958$ 基本相当,二者之差落在 $200$ 个测试样本的采样噪声内,没有出现训练远高于测试的过拟合缺口)。

---

## ④ 研究线(The Where)

卷积网络不是新东西。福岛邦彦的 Neocognitron(1980)已有局部感受野与层级的雏形;LeCun 等人的 **LeNet-5**(1998)把卷积 + 池化 + 反向传播拼成完整流水线,在手写数字上落地成了真实的支票识别系统。但深度学习的引信,是 2012 年 Krizhevsky 等人的 **AlexNet**:在 ImageNet 上,一个八层的 CNN 把 top-5 错误率从上一年的 $\sim 26\%$ 砍到 $15.3\%$,断崖式领先所有基于手工特征的方法。这一击直接把 Ch16 讲过的 SIFT、HOG 那套精心设计的视觉特征扫进了历史——**学到的特征全面碾压了手工的特征**。此后 VGG 把网络加深、**ResNet**(2015)用残差连接把深度推到上百层、把 top-5 错误率进一步压到 $3.57\%$(何恺明同组同年已率先跨过人类约 $5\%$ 的错误率门槛),深度学习的第一波浪潮就此成形。

但请把卷积放在一个更大的图景里看。卷积是一种**硬编码的强归纳偏置**:我们把"平移对称性"这条关于世界的知识,焊死进了架构。它的对立面是 **Transformer / 视觉 Transformer (ViT)**(前指 Ch35):ViT 几乎不假设图像的空间结构,用更弱的先验换取更大的灵活性——代价是它需要**更多的数据**才能自己学出卷积免费送你的那些不变性。这就引出深度学习设计里一个反复出现的、诚实的权衡:

> **强先验 + 小数据,还是弱先验 + 大数据?** 当数据稀缺、结构明确时,把先验烤进架构(卷积)让你用更少的样本学到更好的表示;当数据海量时,弱先验的模型反而能超越你手工设定的那些假设,学到你没想到的结构。

而卷积的平移等变,又只是一个更宏大主题的**特例**:**几何深度学习 (geometric deep learning)** 主张,一切对称性都可以、也应当被翻译成架构约束——平移对称给你卷积,旋转对称给你等变卷积,图上的置换对称给你图神经网络。对称性 → 架构,是一条统一的设计原理。

一个可检验的问题,留给你带着走:**在把先验烤进架构与用数据学出先验之间,临界的数据量在哪里?** 具体地——固定一个视觉任务,一边训 CNN、一边训无卷积先验的 ViT/MLP,画出"测试准确率 vs 训练集大小"两条曲线,它们会在某个数据量交叉吗?交叉点由什么决定?这不是修辞,是一个你今天就能用几张 GPU 跑出来的实验,而它的答案正是当下架构设计辩论的核心。

---

## ⑤ 检索式自测

先自己作答,再展开对照。

**Q1.** 一张 $28\times28$ 的单通道图,过一个 $6$ 个 $5\times5$ 滤波器、valid、步长 $1$ 的卷积层,输出特征图的形状 $(C_{\text{out}}, H_o, W_o)$ 是多少?这一层有多少参数?

**Q2.** 用一句话说清"平移等变"和"平移不变"的区别,并指出 CNN 里哪个组件负责哪个。

**Q3.** 权重共享为什么能推出平移等变?证明的关键一步是什么?

**Q4.** 为什么把图像所有像素按一个固定随机置换打乱后,全连接网络性能不变而 CNN 会崩?

**Q5.** 卷积层参数量为什么与输入图像的尺寸无关?

<details>
<summary>展开答案</summary>

**A1.** $H_o = W_o = 28-5+1 = 24$,故形状为 $(6, 24, 24)$。参数 $= C_{\text{out}}\cdot C_{\text{in}}\cdot k_h\cdot k_w + C_{\text{out}} = 6\cdot1\cdot5\cdot5 + 6 = 156$。

**A2.** 等变:输入平移,输出**以同样方式平移**;不变:输入平移,输出**不动**。卷积层给等变,池化层(以及最终的下采样/读出)累积出局部不变。

**A3.** 因为同一个核 $K(u,v)$ 用在所有位置、不依赖位置 $(i,j)$;证明中把 $(T_\delta X)(i+u,j+v)=X(i+u,j+v-\delta)$ 代入,由于核与位置无关,可把 $\delta$ 干净地提到输出下标上,得到 $S(i,j-\delta)$。若每个位置有独立权重,这一步就失败。

**A4.** 全连接层本就不假设输入顺序(任意两维可任意相关),置换只是重排它权重矩阵的列,可逆且不损失信息,性能不变;CNN 的先验是"相邻像素相关 + 平移有意义",打乱后局部窗口里全是无关像素、平移不再对应任何真实变换,它的强先验变成了错先验。

**A5.** 因为卷积用**同一个** $C_{\text{out}}\times C_{\text{in}}\times k_h\times k_w$ 的核在所有空间位置滑动复用(权重共享),参数只由核大小和通道数决定;图更大只是让核多滑几个位置,不增加任何参数。

</details>

---

## ⑥ 白板自检清单

拿一块白板,不看书,做到以下每一条:

- [ ] 写出离散卷积(互相关)的求和式,并推出 valid 模式下的输出形状 $H_o=H-k_h+1$。
- [ ] 画出 $(N,C,H,W)$ 张量与 $(C_{\text{out}},C_{\text{in}},k_h,k_w)$ 核的形状对应,说清一次卷积是哪些轴在做张量收缩。
- [ ] 只用一行代数证明 $\text{Conv}_K(T_\delta X)=T_\delta(\text{Conv}_K X)$,并指出哪一步用到了权重共享。
- [ ] 说清等变 vs 不变的差别,以及池化如何把前者转成后者、代价是什么(丢失精确位置信息)。
- [ ] 徒手写出卷积反向传播里 $\mathrm{d}W$ 和 $\mathrm{d}x$ 的散射/累加规则,并解释"权重共享 → 所有位置梯度累加到同一核"。
- [ ] 对比全连接层 $O(\text{输入}\times\text{输出})$ 与卷积层 $O(\text{核}\times\text{通道})$ 的参数量,并说明为什么省参数**同时**意味着更强的先验、更低的有效容量。
- [ ] 举一个卷积先验会失灵的数据(打乱像素 / 表格数据),说清它为什么失灵——把 no free lunch 讲成自己的话。

---

## ⑦ 回到中心对象

本书的中心对象始终是同一句:**智能是在不确定性与资源约束下,从经验中获得达成目标能力的过程;而学习,是用数据做归纳推断。**这一章,我们盯住的是"资源约束"这四个字。

MLP 的两宗罪——参数爆炸、无视结构——本质都是**在资源约束下的浪费**:它花掉一千万参数、花掉海量样本,去重新发现"图像有局部性、有平移不变性"这两条我们本就知道的事实。卷积给出的答案,是把这份关于世界的正确先验**直接烤进架构**:局部感受野编码局部性,权重共享编码平移不变。于是同一个能力,用少三个数量级的参数、少得多的数据就学到了,而且泛化更好。这正是"从经验中高效获得能力"最锋利的一次演示——**架构本身,成了你对世界下的那注**。Ch1 教给我们"归纳偏置"这个词,Ch5 把它锐化成"选一个表示就是下一次注";而这一章,那注第一次以钢筋水泥的形态被你亲手实现、亲手训练、亲手核验。

但请注意卷积赌的到底是什么:它赌的是**空间结构**——图像的像素排在一张二维网格上,平移是有意义的对称。这个赌注为图像量身定制。可是世界上另一大类数据,结构根本不是空间的。一句话里的词、一段语音的采样、一条时间序列的读数——它们的结构是**顺序的、变长的**:一个词的含义依赖它前面的词,一个句子可以是五个词也可以是五十个词,而"把第 3 个词平移到第 7 个位置"通常并不保持语义。卷积那套"固定网格 + 平移对称"的先验,在这里是错的偏置。

那么,该为"顺序与变长"这种结构烤进什么样的归纳偏置?答案是一类沿着时间轴共享权重、把过去压缩进一个不断更新的状态里的架构——**循环网络 (recurrent network)**。这就是下一章 Ch27《循环网络与序列》要下的另一注。

---

## 习题

**数学题**

1.(★)推导:输入 $H\times W$,核 $k_h\times k_w$,填充 (padding) $p$,步长 (stride) $s$,证明输出高为 $\big\lfloor (H + 2p - k_h)/s \big\rfloor + 1$。用它验证 "same" 卷积($s=1$)需要 $p=(k-1)/2$。

2.(★★)完整证明二维卷积对**任意二维平移** $(\delta_h,\delta_w)$ 的等变性,并说明为什么它对**旋转 $90^\circ$ 不**等变——由此点明卷积只烤进了平移这一种对称。

3.(★★)承 ⑤ 自测 Q1 的 $28\times28$ 单通道输入与那层 $6$ 个 $5\times5$ 滤波器(参数量 $156$ 见 A1)。现在把它换成一个功能对等的全连接层:把 $784$ 维输入全连接到 $6\cdot24\cdot24$ 维输出,算出它的参数量,并给出与卷积层的比值——用一个具体数字体会"参数与图像尺寸解耦"到底省了多少。

4.(★★★)推导单通道卷积的反向传播:给定上游梯度 $\partial L/\partial S$,写出 $\partial L/\partial K$ 与 $\partial L/\partial X$ 的表达式,并证明"对 $X$ 的梯度等于用翻转后的核对上游梯度做一次(全)卷积"。把它与代码中 `conv_backward` 的散射累加对应起来。

5.(★★)最大池化带来了何种不变性?给出一个具体例子,说明它同时丢失了什么信息(提示:精确定位),并论证这个代价在"图里有没有猫"任务上无害、在"猫在图的哪个坐标"任务上有害。

**编程题**

6.(★)在给定代码上把输入改成 $3$ 通道、卷积滤波器数量从 $4$ 提到 $8$,确认前向/反向形状全部自洽、仍能训练到高准确率。

7.(★★)训练后,把 $4$ 个学到的 $3\times3$ 滤波器 $W1$ 用灰度图可视化;它们是否呈现出横向/纵向的方向选择性?写一段话解释你看到的与任务的关系。

8.(★★★)**让先验失灵**:生成一个固定随机像素置换,对训练集和测试集**同一个**置换地打乱,重训 CNN;再训一个参数量相当的 MLP。报告两者准确率,解释为什么 CNN 的优势蒸发而 MLP 不受影响——把 no free lunch 变成你自己跑出的数字。

9.(★★★)**复现平移鲁棒性对比**:构造一个"条永远居中"的训练集,分别训 CNN 和 MLP;然后在"条随机平移"的测试集上评估。哪个掉得更狠?用平移不变性解释。

**思考题**

10.(★★)开放:给你一个新任务和一个新数据集,你用什么样的判据来决定"该上卷积这种强先验,还是上弱先验 + 更多数据的模型"?列出至少三个你会检查的信号(数据量、是否存在已知对称性、结构是否真为空间性……),并各配一个正反例。

---

## 延伸阅读

- **LeCun, Bottou, Bengio & Haffner (1998), "Gradient-Based Learning Applied to Document Recognition."** LeNet-5 的原始论文——读它看卷积 + 池化 + 反向传播如何第一次拼成一个能上生产线的完整系统,本章代码就是它的最小骨架。
- **Krizhevsky, Sutskever & Hinton (2012), "ImageNet Classification with Deep CNNs."** AlexNet——读它感受那记点燃深度学习浪潮、把手工特征扫进历史的一击,并注意 ReLU、dropout、GPU 这些工程选择的分量。
- **He, Zhang, Ren & Sun (2015), "Deep Residual Learning for Image Recognition."** ResNet——读它理解"如何把网络堆到上百层而不退化",残差连接是后续一切深层架构(包括 Transformer)的地基。
- **Goodfellow, Bengio & Courville,《Deep Learning》第 9 章 "Convolutional Networks."** 教材级的严谨处理——读它把本章的等变、参数共享、池化补齐到形式化的完整论证。
- **Bronstein, Bruna, Cohen & Veličković (2021), "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges."** 几何深度学习纲领——读它把"卷积 = 平移对称的架构化"提升为一条统一原理,看清 CNN 只是"对称性 → 架构"这个大框架里的一个特例,直通 Ch35。