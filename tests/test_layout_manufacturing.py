"""Engineering tests; unit arrays are never manufacturing performance samples."""
import numpy as np
import pytest
from taiwan_lottery.ml.layout.blind.manufacturing import manufacturing_request,score_excel,aggregate,METHODS
from taiwan_lottery.ml.layout.blind.manufacturing_worker import feature_cache
from taiwan_lottery.ml.layout.schema import PredictionRequest
from taiwan_lottery.ml.layout.models import candidate_features
from taiwan_lottery.ml.layout.blind.protocol import save_new


def test_mask_independent_entropy_exactly_80_visible_and_73_hidden():
    values=list(range(1,161))  # Unit contract only, not a manufacturing generator.
    patterns=set()
    for _ in range(100):
        request=manufacturing_request(values)
        assert set(request)=={'rows','cols','cells','target'}
        assert request['target']==73 and request['cells'][72] is None
        assert sum(v is not None for v in request['cells'])==80
        assert 73 not in request['cells']
        patterns.add(tuple(v is not None for v in request['cells']))
    assert len(patterns)==100


def test_feature_cache_bit_exact_for_all_missing_targets_and_new_board():
    cached=feature_cache(candidate_features)
    for values in ([i if i%2 else None for i in range(1,161)], [None]*80+list(range(81,161))):
        b=PredictionRequest(rows=10,cols=16,cells=values,target=2)
        for number in b.missing:assert np.array_equal(cached(b,number),candidate_features(b,number))


def test_scorer_cannot_read_excel_before_seal_and_child_exit(tmp_path,monkeypatch):
    called=[]
    monkeypatch.setattr('taiwan_lottery.ml.layout.blind.manufacturing.frozen_values',lambda p:called.append(p))
    with pytest.raises(FileNotFoundError):score_excel(tmp_path,tmp_path/'answers.xlsx',{})
    assert not called
    save_new(tmp_path/'predictor_exit.json',{'predictor_exited':False,'exitcode':1})
    with pytest.raises(ValueError):score_excel(tmp_path,tmp_path/'answers.xlsx',{})
    assert not called


def test_metrics_count_hits_and_cluster_boards_not_masks():
    rows=[]
    for board,rank in [('a',1),('b',80)]:
        for _ in range(10):
            rows.append({'physical_board_id':board,'unknown_count':80,
                         'metrics':{m:{'actual_rank':rank} for m in METHODS}})
    result=aggregate(rows,bootstrap=100)
    for value in result.values():
        assert value['boards']==2 and value['samples']==20
        assert value['top1_hits']==10 and value['top1']==.5
        assert value['top1_random']==pytest.approx(1/80)
        assert value['top10_random']==pytest.approx(10/80)
        assert value['mean_rank']==40.5 and value['median_rank']==40.5
        assert value['leakage_alert']
        assert 'SUSPECT LEAKAGE' in value['assessment']
