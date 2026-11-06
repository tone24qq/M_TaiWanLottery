import numpy as np
import pytest
from pydantic import ValidationError
from taiwan_lottery.ml.layout.schema import Board, PredictionRequest, HistoryRequest
from taiwan_lottery.ml.layout.predictor import predict
from taiwan_lottery.ml.layout.assignment import assignment, normalize
from taiwan_lottery.ml.layout.consensus import combine
from taiwan_lottery.ml.layout.backtesting import evaluate
from taiwan_lottery.ml.layout.registry import train_models, CLASSES


def test_validation_and_already_revealed():
    for cells in ([1,1,None,None],[0,2,None,None],[5,2,None,None],[1]):
        with pytest.raises(ValidationError):
            Board(rows=2,cols=2,cells=cells)
    with pytest.raises(ValidationError):
        HistoryRequest(rows=2,cols=2,cells=[1,2,3,None],source="partial")
    b=PredictionRequest(rows=2,cols=2,cells=[1,None,None,None],target=1)
    assert predict(b)["status"]=="already_revealed"


def test_no_data_is_honest_and_complete():
    b=PredictionRequest(rows=2,cols=3,cells=[1,None,None,None,5,None],target=3)
    result=predict(b)
    assert result["round_count"]==3
    assert len(result["consensus"]["ranking"])==4
    assert all(x["score"]==.25 for x in result["consensus"]["ranking"])
    assert len([p for p in result["round1"] if p["status"]=="not_ready"])==7
    assert result["consensus"]["strength"]=="no_evidence"
    assert not result["discussion_enabled"]
    assert len(result["consensus"]["assignment"])==4
    assert len({(r["row"],r["col"]) for r in result["consensus"]["assignment"]})==4


def test_joint_assignment_resolves_collisions():
    b=Board(rows=1,cols=3,cells=[None]*3)
    matrix=np.array([[.9,.08,.02],[.8,.15,.05],[.7,.1,.2]])
    a=assignment(b,matrix)
    assert {x["number"] for x in a}=={1,2,3}
    assert len({x["col"] for x in a})==3


def test_identical_models_have_one_cluster_influence():
    a=normalize([[8,2],[2,8]])
    b=normalize([[2,8],[8,2]])
    _,meta=combine(["a","clone","b"],[a,a,b],[])
    w=meta["effective_vote_weight"]
    assert w["a"]+w["clone"]==pytest.approx(w["b"])
    assert meta["weight_mode"]=="equal_history_prior"


def test_history_predictions_reproducible_full_rankings():
    b=PredictionRequest(rows=2,cols=2,cells=[1,None,None,None],target=3)
    history=[[1,2,3,4],[1,3,2,4],[2,1,3,4]]
    a=predict(b,history)
    assert a==predict(b,history)
    for p in a["round1"]:
        if p["status"]=="ready":
            assert len(p["ranking"])==3
            assert len(p["score_matrix"])==3
            assert len(p["assignment"])==3
    for review in a["round2"]:
        assert review["before_ranking"]==review["after_ranking"]


def test_walk_forward_never_trains_on_heldout_and_gain_is_honest():
    rng=np.random.default_rng(5)
    history=[rng.permutation(np.arange(1,7)).tolist() for _ in range(7)]
    report=evaluate(history,2,3,max_folds=6)
    assert len(report["folds"])==6
    for f in report["folds"]:
        assert max(f["training_indices"])<f["heldout_index"]
        assert f["trace"]["training_layout_count"]==len(f["training_indices"])
    assert report["discussion_gain"]==0
    assert report["discussion_improved"] is False
    assert report["individual_models"]["statistical"]["samples"]==6
    assert all("actual_rank" in vote for vote in report["votes"])


@pytest.mark.parametrize("name",["lightgbm","catboost","neural","transformer"])
def test_trained_local_models_produce_finite_complete_scores(name):
    rng=np.random.default_rng(11)
    history=[rng.permutation(np.arange(1,7)).tolist() for _ in range(20)]
    b=PredictionRequest(rows=2,cols=3,cells=[1,None,None,None,None,6],target=2)
    model=next(cls for cls in CLASSES if cls.name==name)()
    model.rows, model.cols=b.rows,b.cols
    model.fit(history)
    assert model.ready, model.reason
    scores=model.scores(b)
    assert scores.shape==(4,4)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(axis=1),1)


def test_api_import_startup_and_validation():
    from fastapi.testclient import TestClient
    from taiwan_lottery.main import app
    with TestClient(app) as c:
        assert c.get("/layout").status_code==200
        assert c.get("/openapi.json").status_code==200
        assert c.post("/api/v1/layout/predict",json={
            "rows":2,"cols":2,"cells":[1,1,None,None],"target":2}).status_code==422
