"""Quadratic Weighted Kappa."""

# TODO: This part references from Taghipour and Ng (2016) heavily. LICENSE???
import numpy as np
# from datasets import meta

score_range = {"1": (2, 12),
        "2": (1, 6),
        "3": (0, 3),
        "4": (0, 3),
        "5": (0, 4),
        "6": (0, 4),
        "7": (0, 30),
        "8": (0, 60)}

def int_scores(scores, lo, hi):
    """TODO"""
    return np.rint(lo+scores*(hi-lo)).astype(np.uint8)

def qwk(scores_infer, scores_gold, domain_id):
    """
        scores_infer: 一组0～1之间的inference分数(一维numpy向量),
        scores_gold: 一组0～1之间的正确分数(一维numpy向量),
        domain_id: domain的id，整数或者字符串都行

    """
    lo, hi = score_range[str(domain_id)]
    num_scores = hi - lo + 1
    num_items = float(len(scores_infer))
    possible_scores = range(lo, hi+1)
    scores_infer = int_scores(scores_infer, lo, hi)
    scores_gold = int_scores(scores_gold, lo, hi)

    # weight matrix
    W = np.zeros([num_scores, num_scores])
    for i in possible_scores:
        for j in possible_scores:
            W[i-lo, j-lo] = ((i-j) / (num_scores-1)) ** 2
    # print(W)

    # confusion matrix
    O = np.zeros([num_scores, num_scores])
    for i, j in zip(scores_gold, scores_infer):
        O[i-lo, j-lo] += 1

    # histogram outerproduct matrix
    histo_infer = np.zeros(num_scores)
    histo_gold = np.zeros(num_scores)
    for s in scores_infer:
        histo_infer[s-lo] += 1
    for s in scores_gold:
        histo_gold[s-lo] += 1
    E = np.outer(histo_gold, histo_infer) / num_items

    numerator = np.sum(np.multiply(W, O))

    denominator = np.sum(np.multiply(W, E))
    if numerator == 0 and denominator == 0:
        return 1
    else:
        kappa = 1 - numerator/denominator
        return kappa
