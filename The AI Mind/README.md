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
| [`03_Self_Evaluation.md`](03_Self_Evaluation.md) | 独立外审自我评估（7 维打分 + 诚实列不足） |
| [`code/`](code/) | 第 1–4 章配套代码 + 回归测试 + 锁定依赖（见 `code/README.md`） |

## 快速运行配套代码

```bash
cd "The AI Mind/code"
pip install -r requirements.txt
python -m ch01.learning_problem      # 复现第 1 章那张过拟合演示表
python -m ch02.faces_of_intelligence # 复现第 2 章 minimax 井字棋 + 组合爆炸
python -m ch03.complexity_wall       # 复现第 3 章 多项式 vs 指数 计数表
python -m ch04.thinking_in_code      # 复现第 4 章 loop vs 向量化 等价与计数
pytest -q                             # 跑全部数值回归测试（ch01–ch04）
```

CI 见 `.github/workflows/aimind-ci.yml`：改动课程代码时自动重跑测试。
