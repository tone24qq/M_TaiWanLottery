"""Empirical reliability, calibration shrinkage and correlated-evidence discount."""
import numpy as np
from scipy.stats import rankdata
from .assignment import normalize

MIN_EVIDENCE = 10


def correlation(matrices):
    vectors = [rankdata(a.ravel()) for a in matrices]
    n = len(vectors)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i):
            x, y = vectors[i], vectors[j]
            if np.std(x) == 0 or np.std(y) == 0:
                value = 1.0 if np.array_equal(x, y) else 0.0
            else:
                value = float(np.corrcoef(x, y)[0, 1])
            corr[i, j] = corr[j, i] = value
    return corr


def reliability(name, records):
    data = [r for r in records if r["model"] == name]
    if len(data) < MIN_EVIDENCE:
        return None
    # All records are prior walk-forward out-of-sample observations of this size.
    def utility(row):
        return np.mean([row["top1"], row["top3"], row["top5"], row["mrr"]])
    full = np.mean([utility(r) for r in data])
    recent = np.mean([utility(r) for r in data[-MIN_EVIDENCE:]])
    brier = np.mean([(r["confidence"] - r["top1"])**2 for r in data])
    stability = np.mean([r.get("stable", 1.0) for r in data])
    return max(1e-6, (full + recent)/2 * (1-brier) * (1+stability)/2)


def combine(names, matrices, evidence):
    corr = correlation(matrices)
    current_correlation = corr.copy()
    historical_pairs = 0
    by_fold = {(r["model"],r["fold"]):r for r in evidence if "fold" in r}
    for i, name in enumerate(names):
        for j in range(i):
            observed = []
            for (model, fold), record in by_fold.items():
                peer = by_fold.get((names[j],fold))
                if model != name or peer is None:
                    continue
                left = {(x["row"],x["col"]):x["score"] for x in record["prediction"]}
                right = {(x["row"],x["col"]):x["score"] for x in peer["prediction"]}
                positions = sorted(left.keys() & right.keys())
                if len(positions)>1:
                    vectors = [np.array([[mapping[p] for p in positions]]) for mapping in (left,right)]
                    observed.append(correlation(vectors)[0,1])
            if len(observed)>=MIN_EVIDENCE:
                corr[i,j] = corr[j,i] = float(np.mean(observed))
                historical_pairs += 1
    # A cluster of identical outputs has total influence of one independent model.
    diversity = 1 / np.maximum(1, np.maximum(corr, 0).sum(axis=1))
    hist = [reliability(name, evidence) for name in names]
    # Equal historical priors until every participating model has enough evidence.
    learned = all(x is not None for x in hist)
    historical = np.asarray(hist if learned else np.ones(len(names)))
    weights = historical * diversity
    weights /= weights.sum()
    calibrated = []
    for name, a in zip(names, matrices):
        past = [r for r in evidence if r["model"] == name]
        if learned:
            mse = np.mean([(r["confidence"]-r["top1"])**2 for r in past])
            a = (1-mse)*a + mse/len(a[0])
        calibrated.append(a)
    score = normalize(sum(w*a for w, a in zip(weights, calibrated)))
    return score, {
        "weight_mode": "walk_forward" if learned else "equal_history_prior",
        "current_prediction_correlation_matrix":current_correlation.tolist(),
        "historical_correlation_pairs":historical_pairs,
        "correlation_models": names, "prediction_correlation_matrix": corr.tolist(),
        "diversity_score": float(np.mean(diversity)),
        "effective_vote_weight": dict(zip(names, map(float, weights))),
        "historical_weight": dict(zip(names, map(float, historical))),
        "calibration": "brier_shrinkage" if learned else "unavailable",
    }
