"""Offline baseline CLI. Never tunes models or rewrites production policy."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from .protocol import partition,masked_request,run_predictor,save_new,observation_time
from .scorer import score_saved,summarize,paired_gains
from ..backtesting import evaluate
from ..registry import CLASSES

BASELINE='28beeb292ecb11dc1dda0d7529f6aae64e80c956'
NAMES=[c.name for c in CLASSES]+['round1','discussion','final_consensus','assignment']


def source_manifest():
    repo=Path(__file__).resolve().parents[4]
    names=['models.py','predictor.py','registry.py','assignment.py','consensus.py','schema.py','backtesting.py']
    files={}
    for name in names:
        relative='taiwan_lottery/ml/layout/'+name
        baseline=subprocess.check_output(['git','show',f'{BASELINE}:{relative}'],cwd=repo)
        current=(repo/relative).read_bytes()
        if current!=baseline:raise ValueError('Baseline model source changed: '+name)
        files[relative]=hashlib.sha256(current).hexdigest()
    return {'baseline_head':BASELINE,'unchanged_predictor_sources_sha256':files,
            'evaluator_sources_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}


def training_bundle(records,rows,cols,shuffle=False):
    history=[list(r['cells']) for r in records if (r['rows'],r['cols'])==(rows,cols)]
    if shuffle:
        rng=np.random.default_rng(2026)
        for h in history:rng.shuffle(h)
    # Evidence only from train-prefix holdouts, never validation/test outcomes.
    prior=evaluate(history,rows,cols,max_folds=min(8,max(0,len(history)-1))) if len(history)>1 else {'votes':[]}
    return {'history':history,'evidence':prior['votes'],'policy_records':prior['votes']}


def execute(corpus,destination,seeds=range(1,21),target=None,walk_forward=True):
    output=Path(destination)
    output.mkdir(parents=True,exist_ok=False)
    manifest=source_manifest();split=partition(corpus)
    seeds=list(seeds)
    if len(set(seeds))!=len(seeds) or not seeds or any(s<0 for s in seeds):raise ValueError('Distinct non-negative seeds required')
    manifest.update(revealed_fraction=.5,hidden_fraction=.5,seeds=seeds,
        physical_boards=len(corpus),partition={k:[r['physical_board_id'] for r in split[k]] for k in ('train','validation','test','unallocated')},
        chronological=split['chronological'],train_validation_test='60/20/20 board split; minimum 5 total boards',
        selection_policy='No fitting, tuning, weighting or model selection from test; train-prefix evidence only',
        isolation='Separate subprocess with allowlisted reads and denied sockets/subprocesses; trusted-code boundary, not hostile-code OS sandbox')
    save_new(output/'protocol.json',manifest)
    all_scores=[];trials=0
    def evaluate_boards(tests,train_provider,protocol):
        nonlocal trials
        scores=[];bundles={}
        for index,record in enumerate(tests):
            train=train_provider(index,record)
            if any(r['physical_board_id']==record['physical_board_id'] or ((r['rows'],r['cols'],r['cells'])==(record['rows'],record['cols'],record['cells'])) for r in train):
                raise ValueError('Training/test board overlap')
            n=record['rows']*record['cols'];targets=range(1,n+1) if target is None else ([target] if target<=n else [])
            for arm in ('baseline','history_label_shuffle','visible_board_shuffle'):
                key=(tuple(r['physical_board_id'] for r in train),record['rows'],record['cols'],arm=='history_label_shuffle')
                if key not in bundles:bundles[key]=training_bundle(train,record['rows'],record['cols'],arm=='history_label_shuffle')
                for number in targets:
                    for seed in seeds:
                        folder=output/'trials'/f'{trials:08d}';trials+=1
                        request=masked_request(record,number,seed,visible_shuffle=arm=='visible_board_shuffle')
                        run_predictor(request,bundles[key],folder)
                        # A separate answer file is created only after prediction
                        # is durably sealed and the predict-only process has exited.
                        answer={**record,'seed':seed,'arm':arm,'protocol':protocol,'visible_shuffle':arm=='visible_board_shuffle'}
                        save_new(folder/'answer.json',answer)
                        score=score_saved(folder,folder/'answer.json');scores.append(score)
                        print(f'{protocol}: {trials} predictions saved',flush=True)
        return scores
    fixed=evaluate_boards(split['test'],lambda i,r:split['train'],'fixed_holdout') if split['test'] else []
    all_scores.extend(fixed)
    walk=[]
    if walk_forward and split['chronological'] and len(corpus)>=5:
        ordered=sorted(corpus,key=lambda r:(observation_time(r),r['physical_board_id']))
        walk=evaluate_boards(ordered[3:],lambda i,r:ordered[:i+3],'walk_forward')
        all_scores.extend(walk)
    baseline=[r for r in fixed if r['arm']=='baseline']
    report={'status':'scored' if baseline else 'not_ready',
        'reason':None if baseline else 'insufficient_real_complete_boards',
        'baseline_source':manifest,'test_boards':len(split['test']),
        'prediction_trials_total_including_controls_and_walk_forward':trials,
        'primary_prediction_trials':len(baseline),'distinct_mask_seeds':len(seeds),
        'overall':summarize(baseline),
        'by_reveal_ratio':{'0.5':summarize(baseline)},
        'by_target':{str(t):summarize([r for r in baseline if r['target']==t]) for t in range(1,max([r['rows']*r['cols'] for r in corpus],default=100)+1)},
        'negative_control_deltas':control_deltas(fixed),
        'negative_controls':{arm:summarize([r for r in fixed if r['arm']==arm]) for arm in ('history_label_shuffle','visible_board_shuffle')},
        'paired_gains':paired_gains(baseline),
        'walk_forward':{'status':'scored' if walk else 'not_ready','reason':None if walk else 'no_dated_sufficient_corpus',
            'overall':summarize([r for r in walk if r['arm']=='baseline']),
            'controls':{arm:summarize([r for r in walk if r['arm']==arm]) for arm in ('history_label_shuffle','visible_board_shuffle')}},
        'missing_models':{name:'not_ready: no scored test trials' for name in NAMES if name not in summarize(baseline)},
        'leakage_checks':{'request_fields_allowlisted':True,'target_forced_hidden':True,'prediction_sealed_before_answer_read':True,
            'train_test_group_disjoint':True,'test_driven_tuning':False,
            'data_dependent_checks_executed':bool(baseline)},
        'assessment':'NO EVIDENCE ABOVE RANDOM' if not baseline else 'See paired board-bootstrap confidence intervals; no automatic promotion',
        'warning':'Repeated masks are dependent. CIs resample physical boards. No synthetic boards used for performance claims.'}
    save_new(output/'blind_baseline.json',report)
    target73=[r for r in baseline if r['target']==73]
    save_new(output/'target_73_report.json',{'status':'scored' if target73 else 'not_ready','samples':len(target73),
        'board_count':len({r['physical_board_id'] for r in target73}),'overall':summarize(target73),
        'trials':[{**r,'model_ranks':{name:m.get('actual_rank') for name,m in r['metrics'].items()}} for r in target73]})
    save_new(output/'all_scores.json',all_scores)
    return report


def control_deltas(scores):
    baseline=[r for r in scores if r['arm']=='baseline']
    result={}
    for arm in ('history_label_shuffle','visible_board_shuffle'):
        lookup={(r['physical_board_id'],r['target'],r['seed']):r for r in scores if r['arm']==arm}
        methods={}
        for name in NAMES:
            pairs=[(r,lookup[(r['physical_board_id'],r['target'],r['seed'])]) for r in baseline
                   if (r['physical_board_id'],r['target'],r['seed']) in lookup]
            pairs=[(a,b) for a,b in pairs if 'actual_rank' in a['metrics'][name] and 'actual_rank' in b['metrics'][name]]
            if not pairs:
                methods[name]={'status':'not_ready'};continue
            methods[name]={key:float(np.mean([a['metrics'][name][key]-b['metrics'][name][key] for a,b in pairs]))
                           for key in ('top1','top3','top5','top10','mrr')}
            methods[name]['interpretation']='Similar control performance can indicate no learned signal or an evaluation problem; it does not itself prove leakage.'
        result[arm]=methods
    return result


async def export_db():
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine
    from taiwan_lottery.config import settings
    from taiwan_lottery.db.models.layout import LayoutHistory
    engine=create_async_engine(settings.DATABASE_URL)
    try:
        async with engine.connect() as connection:
            records=(await connection.execute(select(LayoutHistory.__table__).order_by(LayoutHistory.id))).mappings().all()
            return [{'physical_board_id':r['source'],'rows':r['rows'],'cols':r['cols'],'cells':r['cells'],
                     'provenance':'real_confirmed','observed_at':None,
                     'imported_at':r['created_at'].isoformat(),
                     'time_note':'DB import time is not verified physical-board chronology'} for r in records
                    if not r['source'].lower().startswith(('integration','synthetic','test-','demo'))]
    finally:await engine.dispose()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--corpus',type=Path);source.add_argument('--from-db',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target',type=int,help='Default: every number 1..N, including 73')
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--no-walk-forward',action='store_true')
    args=parser.parse_args()
    if args.target is not None and args.target<1:parser.error('target must be positive')
    if args.seeds<20:parser.error('At least 20 deterministic masks required for the baseline')
    if args.from_db:
        import asyncio
        corpus=asyncio.run(export_db())
    else:corpus=json.loads(args.corpus.read_text())
    execute(corpus,args.output,range(1,args.seeds+1),args.target,not args.no_walk_forward)
