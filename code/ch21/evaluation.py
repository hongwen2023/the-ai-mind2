"""The AI Mind · 第 21 章配套代码：评估、诊断与实验方法

三个演示，把评估的手艺与陷阱跑成可核验的事实：
  一. 从零算混淆矩阵 / precision / recall / F1 / ROC / AUC；
      **两种独立算法核对 AUC**：梯形积分 vs 排序概率定义（应完全一致）。
  二. **揭穿准确率**：极不平衡下，多数类基线准确率≈99% 却召回=0（一文不值）；
      真分类器准确率略低却召回很高。
  三. **k 折交叉验证**：报告各折 + 均值±std（各折有方差 → 单次留出不可靠）。

纯 NumPy、固定种子。
运行 `python -m ch21.evaluation` 打印正文那组可核验输出。
"""
from __future__ import annotations

import numpy as np


def confusion(pred, labels):
    TP = int(np.sum((pred == 1) & (labels == 1)))
    FP = int(np.sum((pred == 1) & (labels == 0)))
    FN = int(np.sum((pred == 0) & (labels == 1)))
    TN = int(np.sum((pred == 0) & (labels == 0)))
    return TP, FP, FN, TN


def prf1(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def roc_auc_trapezoid(scores, labels):
    order = np.argsort(-scores)
    y = labels[order]
    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = np.concatenate([[0], tps / P])
    fpr = np.concatenate([[0], fps / N])
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0)
    return fpr, tpr, float(auc)


def auc_rank_prob(scores, labels):
    """AUC 的排序概率定义：Pr(随机正样本得分 > 随机负样本)，并列算 0.5。"""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    count = 0.0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(count / (len(pos) * len(neg)))


def fit_predict_accuracy(Xtr, Ytr, Xte, Yte):
    mu1 = Xtr[Ytr == 1].mean(0)
    mu0 = Xtr[Ytr == 0].mean(0)
    w = mu1 - mu0
    b = -0.5 * (mu1 @ w + mu0 @ w)
    pred = ((Xte @ w + b) >= 0).astype(int)
    return float(np.mean(pred == Yte))


def kfold_cv(X, Y, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    folds = np.array_split(idx, k)
    accs = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        accs.append(fit_predict_accuracy(X[tr], Y[tr], X[te], Y[te]))
    return np.array(accs)


def run():
    rng = np.random.default_rng(42)

    # 演示一：指标 + AUC 双算法核对
    scores = np.concatenate([rng.normal(1.0, 1.0, 40), rng.normal(-0.3, 1.0, 60)])
    labels = np.concatenate([np.ones(40), np.zeros(60)])
    TP, FP, FN, TN = confusion((scores >= 0.5).astype(int), labels)
    precision, recall, f1 = prf1(TP, FP, FN)
    _, _, auc_trap = roc_auc_trapezoid(scores, labels)
    auc_prob = auc_rank_prob(scores, labels)

    # 演示二：不平衡揭穿准确率
    n = 10000
    y = (rng.random(n) < 0.01).astype(int)
    acc_base = float(np.mean(np.zeros(n, dtype=int) == y))
    rec_base = float(np.sum((np.zeros(n) == 1) & (y == 1)) / max(1, np.sum(y == 1)))
    x = rng.normal(0, 1, n) + y * 3.5
    pred_clf = (x >= 2.2).astype(int)
    acc_clf = float(np.mean(pred_clf == y))
    TP2, FP2, FN2, _ = confusion(pred_clf, y)
    prec_clf, rec_clf, f1_clf = prf1(TP2, FP2, FN2)

    # 演示三：k 折 CV
    N, d = 500, 5
    w_true = rng.normal(0, 1, d)
    X = rng.normal(0, 1, (N, d))
    py = 1 / (1 + np.exp(-(X @ w_true)))
    Y = (rng.random(N) < py).astype(int)
    accs = kfold_cv(X, Y, k=5, seed=7)

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "auc_trap": auc_trap, "auc_prob": auc_prob,
        "n_pos": int(np.sum(y)), "acc_base": acc_base, "rec_base": rec_base,
        "acc_clf": acc_clf, "rec_clf": rec_clf, "f1_clf": f1_clf,
        "cv_accs": accs,
    }


def main() -> None:
    r = run()
    print("precision =", round(r["precision"], 4), " recall =", round(r["recall"], 4),
          " f1 =", round(r["f1"], 4))
    print("AUC trapezoid =", round(r["auc_trap"], 4),
          " rank-prob =", round(r["auc_prob"], 4),
          " diff =", abs(r["auc_trap"] - r["auc_prob"]))
    print(f"\n[imbalance] positives = {r['n_pos']} of 10000")
    print("majority baseline acc =", round(r["acc_base"], 4), "recall =", round(r["rec_base"], 4))
    print("real clf acc =", round(r["acc_clf"], 4), "recall =", round(r["rec_clf"], 4),
          "f1 =", round(r["f1_clf"], 4))
    print("\n[5-fold CV] per fold acc =", np.round(r["cv_accs"], 4))
    print("CV mean =", round(r["cv_accs"].mean(), 4), "std =", round(r["cv_accs"].std(), 4))


if __name__ == "__main__":
    main()
