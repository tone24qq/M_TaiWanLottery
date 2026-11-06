"""Complete rankings and globally consistent one-to-one assignments."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def normalize(matrix):
    a = np.asarray(matrix, dtype=float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Invalid model scores")
    sums = a.sum(axis=1, keepdims=True)
    return np.divide(a, sums, out=np.full_like(a, 1 / a.shape[1]), where=sums > 0)


def ranking(board, scores):
    return [dict(board.coordinate(board.unknown[i]), score=float(scores[i]))
            for i in np.argsort(-np.asarray(scores), kind="stable")]


def assignment(board, matrix):
    rows, cols = linear_sum_assignment(-np.asarray(matrix))
    return [{"number": board.missing[r], **board.coordinate(board.unknown[c]),
             "score": float(matrix[r, c])} for r, c in zip(rows, cols)]
