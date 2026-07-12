"""The AI Mind · 第 5 章配套代码：数据——把世界变成数字

把一张含数值列、类别列、短文本列的混合表格，亲手变成特征矩阵 X（即 Ch1 的 X）：
数值列标准化、类别列 one-hot、短文本词袋，拼接成 (n, d) 张量。

并做关键对照实验——同一个"城市"列，整数编码 vs one-hot 编码——用成对距离矩阵
把"表示会撒谎"变成可打印、可断言的事实：
  - 整数编码：北京↔广州 距离 = 2，北京↔上海 = 1（凭空造出"广州两倍远"的假结构）
  - one-hot 编码：任意两个不同类别的欧氏距离恒 = sqrt(2)（等距、诚实）

纯 NumPy、无随机成分，输出逐位确定。
运行 `python -m ch05.turning_world_into_numbers` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np

AREA = np.array([50.0, 80.0, 65.0, 120.0, 95.0])
BEDROOMS = np.array([1.0, 2.0, 2.0, 3.0, 3.0])
CITY = np.array(["北京", "上海", "北京", "广州", "上海"])
TEXTS = ["近 地铁 精装", "近 公园 毛坯", "近 地铁 毛坯", "近 学校 精装", "近 公园 精装"]
CITIES = np.array(["北京", "上海", "广州"])


def standardize(col):
    return (col - col.mean()) / col.std()


def city_integer_encoding():
    return np.array([np.where(CITIES == c)[0][0] for c in CITY])


def city_onehot_encoding():
    return np.eye(len(CITIES))[city_integer_encoding()]   # 把类别映成标准基向量 e_i


def bag_of_words():
    vocab = sorted(set(w for t in TEXTS for w in t.split()))
    def bow(t):
        v = np.zeros(len(vocab))
        for w in t.split():
            v[vocab.index(w)] += 1
        return v
    return vocab, np.array([bow(t) for t in TEXTS])


def build_feature_matrix():
    """返回 (X, vocab)。X 形状 (5, 11) = 2 数值 + 3 one-hot + 6 词袋。"""
    area_z = standardize(AREA)
    bedrooms_z = standardize(BEDROOMS)
    vocab, text_bow = bag_of_words()
    X = np.hstack([
        area_z.reshape(-1, 1), bedrooms_z.reshape(-1, 1),
        city_onehot_encoding(), text_bow,
    ])
    return X, vocab


def pairwise(M):
    n = M.shape[0]
    return np.array([[np.linalg.norm(M[i] - M[j]) for j in range(n)] for i in range(n)])


def main() -> None:
    X, vocab = build_feature_matrix()
    print("vocab =", vocab)
    print("X.shape =", X.shape)
    print("X =\n", np.round(X, 3))

    city_int = city_integer_encoding()
    d_int = pairwise(city_int.reshape(-1, 1).astype(float))
    d_oh = pairwise(city_onehot_encoding())
    print("\n整数编码成对距离:\n", d_int)
    print("one-hot 成对距离:\n", np.round(d_oh, 4))
    diff_pairs = d_oh[~np.eye(len(CITY), dtype=bool) & (d_oh > 0)]
    print("one-hot 下不同类别距离唯一值:", np.unique(np.round(diff_pairs, 6)), " sqrt(2)=", round(np.sqrt(2), 6))


if __name__ == "__main__":
    main()
