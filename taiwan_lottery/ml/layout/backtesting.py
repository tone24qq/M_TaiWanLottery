"""Walk-forward evaluation: fit prefix, freeze predictions, then score truth."""
import numpy as np
from .schema import PredictionRequest
from .predictor import predict


def metrics(ranking, actual):
    rank = next(i+1 for i,r in enumerate(ranking) if (r["row"],r["col"])==actual)
    n = len(ranking)
    return {"actual_rank": rank, "top1": int(rank<=1), "top3": int(rank<=3),
            "top5": int(rank<=5), "top10": int(rank<=10), "mrr": 1/rank,
            "rank_percentile": (n-rank)/max(1,n-1),
            "random_baseline": 1/n, "random_top3":min(3,n)/n,
            "random_top5":min(5,n)/n, "random_top10":min(10,n)/n,
            "random_mrr":sum(1/k for k in range(1,n+1))/n}


def evaluate(history, rows, cols, seed=42, revealed_fraction=0.5, max_folds=8):
    rng = np.random.default_rng(seed)
    evidence, votes, folds = [], [], []
    # Each board is held out as a whole. All its masks use only earlier boards.
    for index in range(1, min(len(history), max_folds+1)):
        truth = history[index]
        cells = list(truth)
        hidden = sorted(rng.choice(len(truth), max(1, len(truth)-int(len(truth)*revealed_fraction)),
                                   replace=False).tolist())
        for p in hidden:
            cells[p] = None
        target_position = int(rng.choice(hidden))
        target = truth[target_position]
        b = PredictionRequest(rows=rows, cols=cols, cells=cells, target=target)
        result = predict(b, history[:index], evidence, votes)
        actual = (target_position//cols+1, target_position%cols+1)
        report = {}
        for name, ranking in [
            ("ordinary_ensemble",result["ordinary_ensemble"]),
            ("round1_ensemble",result["round1_consensus"]),
            ("discussion_ensemble",result["consensus"]["ranking"])]:
            report[name] = metrics(ranking, actual)
        by_model = {p["model"]: p for p in result["round1"] if p["status"]=="ready"}
        for review in result["round2"]:
            name = review["model"]
            before = metrics(review["before_ranking"], actual)
            after = metrics(review["after_ranking"], actual)
            shadow = metrics(review["shadow_ranking"], actual)
            report[name] = {"round1":before,"round2":after}
            vote = {"model":name, **before, "prediction":review["before_ranking"],
                    "vote":review["before_top1"], "confidence":by_model[name]["ranking"][0]["score"],
                    "final_rank":metrics(result["consensus"]["ranking"],actual)["actual_rank"],
                    "hit":bool(before["top1"]), "discussion_revision":review,
                    "revision_result":after["mrr"]-before["mrr"],
                    "revision_gain":shadow["mrr"]-before["mrr"],
                    "stable":int(not review["changed"]), "fold":index}
            # Truth is appended only after all predictions for this fold are immutable.
            votes.append(vote)
            evidence.append(vote)
        folds.append({"heldout_index":index,"training_indices":list(range(index)),
                      "actual":actual,"target":target,"metrics":report,
                      "trace":result})
    aggregate = {}
    for name in ("ordinary_ensemble","round1_ensemble","discussion_ensemble"):
        if folds:
            aggregate[name] = {key:float(np.mean([f["metrics"][name][key] for f in folds]))
                               for key in folds[0]["metrics"][name]}
    individuals = {}
    for name in sorted({v["model"] for v in votes}):
        records = [v for v in votes if v["model"] == name]
        individuals[name] = {"samples":len(records), **{
            key:float(np.mean([v[key] for v in records]))
            for key in ("top1","top3","top5","top10","mrr","rank_percentile")}}
    gain = (aggregate["discussion_ensemble"]["mrr"]-aggregate["round1_ensemble"]["mrr"]) if folds else None
    return {"status":"ready" if folds else "not_ready","seed":seed,"folds":folds,
            "votes":votes,"aggregate":aggregate,"individual_models":individuals,"discussion_gain":gain,
            "discussion_improved":bool(gain is not None and gain>0),
            "protocol":"chronological_board_holdout_walk_forward",
            "warning":"Small samples are not evidence of general predictive accuracy"}
