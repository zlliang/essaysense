import numpy as np

# Score range metadata.
score_range = {"1": (2, 12),
               "2": (1, 6),
               "3": (0, 3),
               "4": (0, 3),
               "5": (0, 4),
               "6": (0, 4),
               "7": (0, 30),
               "8": (0, 60)}

def int_scores(scores, lo, hi):
    """Convert scores ranging in [0, 1] to appropriate integers.

    This function perform a simple linear transformation and rounding process
    to convert scores ranging in [0, 1] to appropriate integers.

    Args:
        - scores: given scores ranging in [0, 1], as a NumPy array.
        - lo, hi: range of the discrete integer values.

    Returns:
        - Integer scores in a NumPy array (type: numpy.uint8).
    """
    return np.rint(lo+scores*(hi-lo)).astype(np.uint8)

def qwk(scores_infer, scores_gold, domain_id):
    """Calculate Quadratic Weighted Kappa value.

    This function calculates the quadratic weighted kappa value, which is a
    measure of inter-rater agreement between two raters that provide discrete
    numeric ratings. Note that for this version, the function is only
    compatible scores from with ASAP-AES dataset.

    Reference:
        Evaluation metrics delivered by ASAP-AES, see:
        https://github.com/benhamner/ASAP-AES.

    Args:
        - scores_infer: prediction scores ranging in [0, 1], as a NumPy array.
        - scores_gold: gold scores ranging in [0, 1], as a NumPy array.
        - domain_id: Prompt ID of the essay dataset.

    Returns:
        - QWK value, potentially ranging in [-1, 1]. The more the value is close
          to 1, the better the prediction scores are.
    """
    lo, hi = score_range[str(domain_id)]
    num_scores = hi - lo + 1
    num_items = float(len(scores_infer))
    possible_scores = range(lo, hi+1)
    scores_infer = int_scores(scores_infer, lo, hi)
    scores_gold = int_scores(scores_gold, lo, hi)

    # Weight matrix.
    W = np.zeros([num_scores, num_scores])
    for i in possible_scores:
        for j in possible_scores:
            W[i-lo, j-lo] = ((i-j) / (num_scores-1)) ** 2

    # Confution matrix
    O = np.zeros([num_scores, num_scores])
    for i, j in zip(scores_gold, scores_infer):
        O[i-lo, j-lo] += 1

    # Histogram outerproduct matrix
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
        return 1  # Prevent overflow.
    else:
        kappa = 1 - numerator/denominator
        return kappa
