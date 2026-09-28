"""Risk grading helpers used by the strategy notebooks."""
import numpy as np

GRADE_LABELS = list("ABCDE")


def grade_cuts(ref_scores, shares):
    """Score cut points (descending) so that the reference population splits into `shares`
    from the safest grade to the riskiest. Scores must be higher = safer."""
    return [float(np.quantile(ref_scores, 1 - c)) for c in np.cumsum(shares)[:-1]]


def to_grade(scores, cuts):
    """Integer grade 0 (A, safest) ... len(cuts) (riskiest)."""
    g = np.full(len(scores), len(cuts), dtype=int)
    for i, c in reversed(list(enumerate(cuts))):
        g[scores >= c] = i
    return g
