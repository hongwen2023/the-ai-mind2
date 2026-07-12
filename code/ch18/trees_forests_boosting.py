"""The AI Mind · 第 18 章配套代码：树、森林与提升

从零实现 CART 决策树（分类 Gini / 回归方差），并演示三件事：
  一. 单棵深树过拟合：train 误差≈0、test 误差高（高方差）；
  二. 随机森林（bagging + 特征子采样）降方差：test 误差 < 单棵深树；
  三. 梯度提升（平方损失=拟合残差=函数空间梯度下降）降偏差：test_mse 随轮数下降。

纯 NumPy、固定种子。
运行 `python -m ch18.trees_forests_boosting` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "gain", "n_samples")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.gain = 0.0
        self.n_samples = 0


class DecisionTree:
    def __init__(self, task="clf", max_depth=None, min_samples_leaf=1,
                 max_features=None, random_state=0):
        self.task = task
        self.max_depth = max_depth if max_depth is not None else 10 ** 9
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

    def _impurity(self, y):
        if self.task == "clf":
            _, counts = np.unique(y, return_counts=True)
            p = counts / counts.sum()
            return 1.0 - np.sum(p * p)      # Gini
        return np.var(y) if len(y) else 0.0  # 回归方差

    def _leaf_value(self, y):
        if self.task == "clf":
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]
        return float(np.mean(y))

    def _best_split(self, X, y):
        n, d = X.shape
        parent_imp = self._impurity(y)
        best_gain, best = 0.0, None
        if self.max_features is not None and self.max_features < d:
            feats = self.rng.choice(d, size=self.max_features, replace=False)
        else:
            feats = np.arange(d)
        for j in feats:
            xj = X[:, j]
            order = np.argsort(xj, kind="mergesort")
            xs, ys = xj[order], y[order]
            for i in range(1, n):
                if xs[i] == xs[i - 1]:
                    continue
                if i < self.min_samples_leaf or (n - i) < self.min_samples_leaf:
                    continue
                yl, yr = ys[:i], ys[i:]
                imp = (i / n) * self._impurity(yl) + ((n - i) / n) * self._impurity(yr)
                gain = parent_imp - imp
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (j, 0.5 * (xs[i] + xs[i - 1]))
        return best, best_gain

    def _build(self, X, y, depth):
        node = Node()
        node.n_samples = len(y)
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf \
                or self._impurity(y) <= 1e-12:
            node.value = self._leaf_value(y)
            return node
        split, gain = self._best_split(X, y)
        if split is None or gain <= 1e-12:
            node.value = self._leaf_value(y)
            return node
        j, t = split
        mask = X[:, j] <= t
        node.feature, node.threshold, node.gain = j, t, gain
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._build(np.asarray(X, float), np.asarray(y), 0)
        return self

    def _predict_one(self, x, node):
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, float)
        return np.array([self._predict_one(x, self.root) for x in X])


def make_clf(n, rng):
    X = rng.uniform(-3, 3, size=(n, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 4.0).astype(int)
    flip = rng.random(n) < 0.10
    return X, np.where(flip, 1 - y, y)


def make_reg(n, rng):
    X = rng.uniform(-3, 3, size=(n, 1))
    y = np.sin(1.5 * X[:, 0]) + 0.3 * X[:, 0] + rng.normal(0, 0.15, size=n)
    return X, y


def err_clf(yhat, y):
    return float(np.mean(yhat != y))


def mse(yhat, y):
    return float(np.mean((yhat - y) ** 2))


def random_forest_fit(X, y, n_trees, max_features, seed=0):
    n = len(y)
    master = np.random.default_rng(seed)
    trees = []
    for _ in range(n_trees):
        idx = master.integers(0, n, size=n)
        tb = DecisionTree(task="clf", min_samples_leaf=1, max_features=max_features,
                          random_state=master.integers(1 << 31)).fit(X[idx], y[idx])
        trees.append(tb)
    return trees


def rf_predict(trees, X):
    votes = np.stack([t.predict(X) for t in trees])
    return (votes.mean(axis=0) >= 0.5).astype(int)


def gbrt_fit(X, y, n_rounds, lr, max_depth, seed=0):
    F0 = float(np.mean(y))
    trees, F = [], np.full(len(y), F0)
    for m in range(n_rounds):
        resid = y - F                              # 负梯度 = 残差（平方损失）
        tm = DecisionTree(task="reg", max_depth=max_depth,
                          min_samples_leaf=5, random_state=seed + m).fit(X, resid)
        F = F + lr * tm.predict(X)
        trees.append(tm)
    return F0, trees


def gbrt_predict(F0, trees, X, lr):
    out = np.full(len(X), F0)
    for t in trees:
        out = out + lr * t.predict(X)
    return out


def run():
    rng = np.random.default_rng(42)
    Xtr, ytr = make_clf(400, rng)
    Xte, yte = make_clf(2000, rng)

    deep = DecisionTree(task="clf", min_samples_leaf=1, random_state=0).fit(Xtr, ytr)
    tree_train = err_clf(deep.predict(Xtr), ytr)
    tree_test = err_clf(deep.predict(Xte), yte)

    forest = random_forest_fit(Xtr, ytr, n_trees=200, max_features=1, seed=7)
    forest_test = err_clf(rf_predict(forest, Xte), yte)

    Rtr_X, Rtr_y = make_reg(300, np.random.default_rng(1))
    Rte_X, Rte_y = make_reg(3000, np.random.default_rng(2))
    F0, gtrees = gbrt_fit(Rtr_X, Rtr_y, n_rounds=100, lr=0.1, max_depth=3, seed=0)
    boost = {M: mse(gbrt_predict(F0, gtrees[:M], Rte_X, 0.1), Rte_y) for M in (1, 5, 20, 100)}

    return {"tree_train": tree_train, "tree_test": tree_test,
            "forest_test": forest_test, "boost_test_mse": boost}


def main() -> None:
    r = run()
    print(f"[单棵深树]     train_err={r['tree_train']:.3f}   test_err={r['tree_test']:.3f}")
    print(f"[随机森林 200] test_err={r['forest_test']:.3f}   (对比单树 test_err={r['tree_test']:.3f})")
    for M, m in r["boost_test_mse"].items():
        print(f"[梯度提升 {M:3d} 轮] test_mse={m:.3f}")


if __name__ == "__main__":
    main()
