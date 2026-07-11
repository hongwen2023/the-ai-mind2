# The AI Mind

一套长期、开源、第一性原理的 AI 课程 —— **Track A（Claude 独立设计赛道）** 首批交付物。

> 中心对象：**智能 = 在不确定性与资源约束下从经验中获得达成目标的能力；学习 = 用数据做归纳推断。**
> 全书 57 个单元（Prelude + 7 本书）都是这一句的展开，用认知依赖图（而非线性目录）排序，
> 数学 / 工程 / 研究三条线在每一章内交织。

## 交付物

| 文件 | 内容 |
|---|---|
| [`00_Architecture.md`](00_Architecture.md) | 课程总架构 v2.0：设计哲学、书目结构、依赖图、三线设计、为什么最优、可维护性、18 条决策记录 |
| [`01_Prelude.md`](01_Prelude.md) | Prelude P.1《为什么学习 AI 需要一张地图》 |
| [`02_Chapter01_The_Learning_Problem.md`](02_Chapter01_The_Learning_Problem.md) | 第 1 章《学习问题——经验如何变成能力》（含可运行 NumPy 实验、习题） |
| [`04_Chapter02_Faces_of_Intelligence.md`](04_Chapter02_Faces_of_Intelligence.md) | 第 2 章《智能的多副面孔》（搜索/逻辑/学习三范式，含 minimax 井字棋实验、习题） |
| [`05_Chapter03_Power_and_Cost_of_Computation.md`](05_Chapter03_Power_and_Cost_of_Computation.md) | 第 3 章《计算的能力与代价》（图灵机/停机问题/P vs NP/多项式 vs 指数，含复杂度实验、习题） |
| [`06_Chapter04_Programming_as_a_Tool_for_Thought.md`](06_Chapter04_Programming_as_a_Tool_for_Thought.md) | 第 4 章《编程作为思维工具》（张量/向量化/广播，含 loop vs vectorized 实测、习题） |
| [`07_Chapter05_Turning_the_World_into_Numbers.md`](07_Chapter05_Turning_the_World_into_Numbers.md) | 第 5 章《数据——把世界变成数字》（表示/one-hot/词袋/i.i.d.，含整数 vs one-hot 距离对照、习题） |
| [`08_Chapter06_From_Samples_to_Rules.md`](08_Chapter06_From_Samples_to_Rules.md) | 第 6 章《从样本到规律——归纳第一课》（欠定/奥卡姆/记忆vs泛化/训练-验证-测试，含 kNN 实测、习题） |
| [`09_Chapter07_Vertical_Slice.md`](09_Chapter07_Vertical_Slice.md) | 第 7 章《垂直切片——先跑通一个会学习的系统》（Book I 收官：从零 logistic regression + 梯度下降 + 梯度检验） |
| [`10_Chapter08_Geometry_of_Linear_Algebra.md`](10_Chapter08_Geometry_of_Linear_Algebra.md) | 第 8 章《线性代数——表示与变换的几何》（Book II 开篇：矩阵=变换/秩/投影/SVD/低秩逼近） |
| [`11_Chapter09_Matrix_Calculus_and_Numerics.md`](11_Chapter09_Matrix_Calculus_and_Numerics.md) | 第 9 章《矩阵微分与数值基础》（梯度/雅可比/链式法则；**严格推导偿还 Ch7 梯度欠条**；log-sum-exp/条件数） |
| [`12_Chapter10_Probability.md`](12_Chapter10_Probability.md) | 第 10 章《概率论——不确定性的演算》（随机变量/期望/贝叶斯/似然；**集中不等式 Hoeffding**——通往泛化界的桥） |
| [`13_Chapter11_Statistical_Inference.md`](13_Chapter11_Statistical_Inference.md) | 第 11 章《统计推断——从数据到结论》（MLE/贝塞尔校正/MAP=正则化；**偏差-方差分解**——把 U 型曲线严格拆开） |
| [`14_Chapter12_Information_Theory.md`](14_Chapter12_Information_Theory.md) | 第 12 章《信息论——惊奇、熵与压缩》（熵/交叉熵/KL；**命名定理：min 交叉熵 ⇔ 最大似然 ⇔ min KL**——交叉熵合法性总闸门） |
| [`15_Chapter13_Formalizing_Learning_ERM.md`](15_Chapter13_Formalizing_Learning_ERM.md) | 第 13 章《学习问题的形式化——损失、风险与 ERM》（真实风险/经验风险/ERM/**贝叶斯风险地板**/近似-估计误差分解/代理损失） |
| [`16_Chapter14_Optimization.md`](16_Chapter14_Optimization.md) | 第 14 章《优化——如何下山》（凸性/**下降引理偿还 Ch7 GD 收敛欠条**/收敛率/SGD/非凸桥/KKT） |
| [`17_Chapter15_Learning_Theory.md`](17_Chapter15_Learning_Theory.md) | 第 15 章《学习理论——泛化为何可能》（Book II 收官：**泛化界=Hoeffding+union bound**/PAC/VC 维/经典理论边界） |
| [`18_Chapter16_Linear_Models.md`](18_Chapter16_Linear_Models.md) | 第 16 章《线性模型》（Book III 开篇：机器学习的"氢原子"；线性/逻辑回归/L2 岭/L1 lasso/正则化路径） |
| [`19_Chapter17_Kernels_and_SVM.md`](19_Chapter17_Kernels_and_SVM.md) | 第 17 章《核方法与 SVM》（核技巧/RBF 无穷维/表示定理/**SVM 对偶推导+KKT 支持向量**/软间隔 hinge） |
| [`20_Chapter18_Trees_Forests_Boosting.md`](20_Chapter18_Trees_Forests_Boosting.md) | 第 18 章《树、森林与提升》（决策树/Gini/**装袋降方差+随机森林去相关**/**梯度提升=函数梯度下降**） |
| [`03_Self_Evaluation.md`](03_Self_Evaluation.md) | 独立外审自我评估（7 维打分 + 诚实列不足） |
| [`code/`](code/) | 第 1–18 章配套代码 + 回归测试 + 锁定依赖（见 `code/README.md`） |

> **Book I（Ch1–7）与 Book II（Ch8–15）均已完整交付；Book III（经典机器学习，Ch16–21）进行中。** Book II 收官时 **Ch7 的四张欠条全部结清**：梯度推导(Ch9)、交叉熵合法性(Ch12)、GD 收敛(Ch14)、泛化界(Ch15)。Prelude P.1、架构、自评见上表。

## 快速运行配套代码

```bash
cd "The AI Mind/code"
pip install -r requirements.txt
python -m ch01.learning_problem      # 复现第 1 章那张过拟合演示表
python -m ch02.faces_of_intelligence # 复现第 2 章 minimax 井字棋 + 组合爆炸
python -m ch03.complexity_wall       # 复现第 3 章 多项式 vs 指数 计数表
python -m ch04.thinking_in_code      # 复现第 4 章 loop vs 向量化 等价与计数
python -m ch05.turning_world_into_numbers  # 复现第 5 章 整数 vs one-hot 距离对照
python -m ch06.induction_first_lesson      # 复现第 6 章 kNN 训练/测试误差表
python -m ch07.vertical_slice              # 复现第 7 章 logistic regression + 梯度检验
python -m ch08.geometry_of_linear_algebra  # 复现第 8 章 SVD 椭圆 + Eckart–Young
python -m ch09.matrix_calculus_and_numerics # 复现第 9 章 梯度检验 + 稳定softmax + 条件数
python -m ch10.probability                 # 复现第 10 章 蒙特卡洛 + Hoeffding 经验验证
python -m ch11.bias_variance               # 复现第 11 章 偏差-方差分解表
python -m ch12.information_theory           # 复现第 12 章 熵/CE/KL + 命名定理
python -m ch13.erm_and_bayes_risk           # 复现第 13 章 贝叶斯地板 + 风险分解
python -m ch14.optimization                 # 复现第 14 章 GD 步长三态 + SGD
python -m ch15.learning_theory              # 复现第 15 章 泛化差距 vs n/容量/随机标签
python -m ch16.linear_models                # 复现第 16 章 岭/lasso 正则化路径
python -m ch17.kernels_and_svm              # 复现第 17 章 线性vs RBF核 + 支持向量
python -m ch18.trees_forests_boosting       # 复现第 18 章 单树/森林/梯度提升
pytest -q                             # 跑全部数值回归测试（ch01–ch18，83 项）
```

CI 见 `.github/workflows/aimind-ci.yml`：改动课程代码时自动重跑测试。
