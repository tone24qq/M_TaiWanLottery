"""Synthetic contract fixtures only; none are evidence of real model accuracy."""
import inspect
import json
from pathlib import Path
import subprocess
import sys
import pytest
from taiwan_lottery.ml.layout.blind.protocol import masked_request,partition,run_predictor,save_new,digest
from taiwan_lottery.ml.layout.blind.worker import predict_request
from taiwan_lottery.ml.layout.blind.scorer import score_saved,summarize,paired_gains
from taiwan_lottery.ml.layout.blind.run import execute,source_manifest


def fixture_board(i=0):
    # Fabricated permutation for schema/leakage testing, never benchmark input.
    cells=list(range(1,101));cells=cells[i:]+cells[:i]
    return {'rows':10,'cols':10,'cells':cells,'physical_board_id':f'unit-{i}',
            'provenance':'real_confirmed'}


def test_target_always_hidden_exact_half_and_request_contract():
    b=fixture_board()
    for target in range(1,101):
        for seed in range(1,21):
            request=masked_request(b,target,seed)
            assert set(request)=={'rows','cols','cells','target'}
            assert sum(v is None for v in request['cells'])==50
            assert request['cells'][b['cells'].index(target)] is None
            assert target not in request['cells']
            assert request==masked_request(b,target,seed)
            assert all(v is None or v==b['cells'][i] for i,v in enumerate(request['cells']))
    assert list(inspect.signature(predict_request).parameters)==['request','training']
    with pytest.raises(ValueError):predict_request({**request,'full_board':b['cells']},{})


def test_group_split_rejects_duplicate_boards_and_untrusted_provenance():
    rows=[fixture_board(i) for i in range(10)]
    split=partition(rows)
    assert [len(split[k]) for k in ('train','validation','test')]==[6,2,2]
    assert len({r['physical_board_id'] for k in ('train','validation','test') for r in split[k]})==10
    with pytest.raises(ValueError):partition([rows[0],{**rows[1],'physical_board_id':rows[0]['physical_board_id']}])
    with pytest.raises(ValueError):partition([rows[0],{**rows[0],'physical_board_id':'alias'}])
    with pytest.raises(ValueError):partition([{**rows[0],'provenance':'synthetic'}])
    assert not partition(rows[:4])['test']


def test_persist_then_score_and_prevent_answer_or_prediction_rewrite(tmp_path):
    request={'rows':2,'cols':2,'cells':[None,2,None,4],'target':1}
    training={'history':[],'evidence':[],'policy_records':[]}
    folder=tmp_path/'trial'
    answer=tmp_path/'secret.json'
    save_new(answer,{'rows':2,'cols':2,'cells':[1,2,3,4],'physical_board_id':'unit-only',
                     'seed':1,'arm':'baseline','protocol':'unit-test'})
    with pytest.raises(FileNotFoundError):score_saved(folder,answer)
    result=run_predictor(request,training,folder)
    assert result['methods']['statistical']['status']=='ready'
    assert len(result['methods']['statistical']['ranking'])==2
    before=digest(folder/'prediction.json')
    score=score_saved(folder,answer)
    assert score['random']['top1']==.5
    assert score['random']['top3']==1
    assert score['metrics']['statistical']['actual_rank']==1
    assert digest(folder/'prediction.json')==before
    with pytest.raises(FileExistsError):save_new(folder/'prediction.json',{})
    with pytest.raises(FileExistsError):score_saved(folder,answer)
    path=folder/'prediction.json';path.chmod(0o644);path.write_text('{}')
    with pytest.raises(ValueError,match='modified'):score_saved(folder,answer)


def test_worker_denies_ground_truth_reads_and_network(tmp_path):
    secret=tmp_path/'ground_truth.json';secret.write_text('[73]')
    code='''
import sys,socket
from pathlib import Path
from taiwan_lottery.ml.layout.blind.worker import install_read_guard
install_read_guard([])
try:Path(sys.argv[1]).read_text()
except PermissionError:print('denied truth')
else:raise AssertionError('truth read')
try:socket.create_connection(('127.0.0.1',5432))
except PermissionError:print('denied network')
else:raise AssertionError('network allowed')
'''
    result=subprocess.run([sys.executable,'-c',code,str(secret)],text=True,capture_output=True,check=True)
    assert result.stdout.splitlines()==['denied truth','denied network']


def test_board_bootstrap_and_random_depend_on_unknown_count():
    scores=[]
    for board,n,rank in [('a',2,1),('b',10,5)]:
        for seed in range(20):
            metric={'top1':int(rank==1),'top3':int(rank<=3),'top5':1,'top10':1,'mrr':1/rank,'actual_rank':rank}
            scores.append({'physical_board_id':board,'metrics':{'round1':metric,'discussion':metric,'final_consensus':metric,'assignment':metric},
                'random':{'top1':1/n,'top3':min(3,n)/n,'top5':min(5,n)/n,'top10':1,'mrr':sum(1/i for i in range(1,n+1))/n,'mean_rank':(n+1)/2}})
    report=summarize(scores,100)
    assert report['round1']['board_count']==2
    assert report['round1']['samples']==40
    assert report['round1']['top1_random']==pytest.approx(.3)
    assert report['round1']['mean_rank']==3
    assert report['round1']['median_rank']==3
    assert paired_gains(scores)['discussion_effectiveness']=='discussion effectiveness not established'
    assert paired_gains(scores)['assignment']['net_gain']==0
    assert not paired_gains(scores)['assignment']['force_assignment_recommended']


def test_visible_shuffle_preserves_mask_and_target_exclusion():
    b=fixture_board()
    a=masked_request(b,73,1);z=masked_request(b,73,1,visible_shuffle=True)
    assert [i for i,v in enumerate(a['cells']) if v is None]==[i for i,v in enumerate(z['cells']) if v is None]
    assert sorted(v for v in a['cells'] if v is not None)==sorted(v for v in z['cells'] if v is not None)
    assert a!=z
    assert 73 not in z['cells']


def test_empty_real_corpus_is_not_a_fabricated_benchmark(tmp_path):
    report=execute([],tmp_path/'empty')
    assert report['status']=='not_ready'
    assert report['test_boards']==0
    assert report['primary_prediction_trials']==0
    assert report['overall']['random']=={'status':'not_ready','samples':0}
    assert report['assessment']=='NO EVIDENCE ABOVE RANDOM'
    assert not report['leakage_checks']['data_dependent_checks_executed']
    assert not json.loads((tmp_path/'empty'/'target_73_report.json').read_text())['trials']
    assert source_manifest()['baseline_head']=='28beeb292ecb11dc1dda0d7529f6aae64e80c956'


def test_all_eight_model_paths_in_isolated_worker_on_unit_fixtures(tmp_path):
    """Exercise optional model branches, NOT a real-data accuracy benchmark."""
    import importlib.util
    import itertools
    if any(importlib.util.find_spec(p) is None for p in ('torch','lightgbm','catboost')):
        pytest.skip('optional learned-model dependencies unavailable')
    history=[list(p) for p in itertools.permutations(range(1,5)) if p!=(1,2,3,4)][:20]
    request={'rows':2,'cols':2,'cells':[None,2,None,4],'target':1}
    result=run_predictor(request,{'history':history,'evidence':[],'policy_records':[]},tmp_path/'models')
    for name in ('statistical','position','association','lightgbm','catboost','similarity_knn','neural','transformer'):
        assert result['methods'][name]['status']=='ready'
        assert len(result['methods'][name]['ranking'])==2


def test_chronology_requires_verified_timezone_and_orders_actual_instants():
    rows=[fixture_board(i) for i in range(5)]
    for i,r in enumerate(rows):r['observed_at']=f'2026-01-0{i+1}T00:00:00+00:00'
    split=partition(list(reversed(rows)))
    assert split['chronological']
    assert split['train'][0]['physical_board_id']=='unit-0'
    assert split['test'][0]['physical_board_id']=='unit-4'
    rows[0]['observed_at']='2026-01-01T00:00:00'
    with pytest.raises(ValueError,match='timezone'):partition(rows)
