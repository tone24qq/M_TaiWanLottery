"""Three bounded rounds. No truth input exists in the prediction contract."""
import numpy as np
from .registry import train_models
from .assignment import normalize, ranking, assignment
from .consensus import combine, MIN_EVIDENCE


def proposal(board, model, matrix):
    r = ranking(board, matrix[board.missing.index(board.target)])
    confidence = float(np.clip((matrix[board.missing.index(board.target)].max()*len(board.unknown)-1) /
                       max(1, len(board.unknown)-1), 0, 1))
    return {"model": model.name, "status": "ready", "top1": r[0],
            "top3": r[:3], "top5": r[:5], "ranking": r,
            "confidence": confidence, "confidence_kind": "score_concentration_not_hit_probability",
            "features": model.features, "reasons": model.features,
            "score_matrix": matrix.tolist(), "assignment": assignment(board, matrix)}


def predict(board, history=(), evidence=(), policy_records=()):
    if board.target in board.cells:
        i = board.cells.index(board.target)
        return {"status": "already_revealed", "target": board.target, **board.coordinate(i)}
    models = train_models(board, history)
    active = [m for m in models if m.ready]
    # Independent scoring completes before any model sees a peer output.
    matrices = [normalize(m.scores(board)) for m in active]
    names = [m.name for m in active]
    before = [proposal(board, m, a) for m, a in zip(active, matrices)]
    unavailable = [{"model": m.name, "status": "not_ready", "reason": m.reason,
                    "required_complete_layouts": m.minimum, "available_complete_layouts": len(history)}
                   for m in models if not m.ready]
    first, metadata = combine(names, matrices, evidence)
    simple = normalize(np.mean(matrices, axis=0))
    revised, reviews = [], []
    for index, (m, a, p) in enumerate(zip(active, matrices, before)):
        prior = [r for r in policy_records if r["model"] == m.name]
        gain = np.mean([r["revision_gain"] for r in prior]) if prior else 0.0
        # Only earlier shadow holdouts can enable peer influence.
        sensitivity = float(np.clip(gain, 0, 1)) if len(prior) >= MIN_EVIDENCE else 0.0
        others = [x for j, x in enumerate(matrices) if j != index]
        peer = normalize(np.mean(others, axis=0)) if others else a
        candidate = normalize((a + peer)/2)  # fixed shadow protocol, separately evaluated
        after = normalize((1-sensitivity)*a + sensitivity*peer) if sensitivity else a.copy()
        q = proposal(board, m, after)
        shadow = ranking(board, candidate[board.missing.index(board.target)])
        changed = p["ranking"] != q["ranking"]
        reviews.append({"model": m.name, "before_ranking": p["ranking"],
                        "after_ranking": q["ranking"], "shadow_ranking": shadow,
                        "changed": changed, "sensitivity": sensitivity,
                        "disagreement_reasons": ["peer_evidence"] if changed else
                            ["peer_review_not_validated" if not sensitivity else "stable"],
                        "before_top1": p["top1"], "after_top1": q["top1"]})
        revised.append(after)
    final, final_meta = combine(names, revised, evidence)
    joint = assignment(board, final)
    target_index = board.missing.index(board.target)
    r = ranking(board, final[target_index])
    top = (r[0]["row"], r[0]["col"])
    supporting = [p["model"] for p in before if (p["top1"]["row"], p["top1"]["col"]) == top]
    opposing = [p["model"] for p in before if p["model"] not in supporting]
    uniform = bool(np.allclose(final[target_index], final[target_index, 0]))
    disagreement = len(opposing)/len(active)
    assigned = next(x for x in joint if x["number"] == board.target)
    return {
        "status": "ready", "algorithm_version": "layout-v1", "target": board.target,
        "round_count": 3, "training_layout_count": len(history),
        "score_kind": "relative_score_not_hit_probability",
        "number_order": board.missing, "cell_order": [board.coordinate(i) for i in board.unknown],
        "round1": before + unavailable, "round2": reviews,
        "round1_consensus": ranking(board, first[target_index]),
        "ordinary_ensemble": ranking(board, simple[target_index]),
        "consensus": {"ranking": r, "top1": r[0], "top3": r[:3], "top5": r[:5],
                      "score_matrix": final.tolist(), "assignment": joint,
                      "target_assignment": assigned,
                      "assignment_disagrees_with_top1": (assigned["row"], assigned["col"]) != top,
                      "strength": "no_evidence" if uniform or len(history) < 3 else "unvalidated",
                      "disagreement_level": disagreement,
                      "support_models": supporting, "oppose_models": opposing, **final_meta},
        "random_baseline": {"top1": 1/len(r), "top3": min(3,len(r))/len(r),
                            "top5": min(5,len(r))/len(r)},
        "devils_advocate": {
            "candidate": r[0], "support_models": len(supporting), "oppose_models": len(opposing),
            "alternative": r[1] if len(r)>1 else None,
            "risk": "high" if uniform or len(supporting)<=1 or len(history)<3 else "unvalidated",
            "reasons": [code for code, flag in [
                ("insufficient_history", len(history)<20), ("single_model_support",len(supporting)<=1),
                ("uniform_scores",uniform), ("model_disagreement",bool(opposing)),
                ("joint_assignment_conflict",(assigned["row"],assigned["col"])!=top)] if flag]},
        "discussion_enabled": any(x["sensitivity"]>0 for x in reviews),
        "discussion_effectiveness": "not_established",
        "round1_metadata": metadata,
    }
