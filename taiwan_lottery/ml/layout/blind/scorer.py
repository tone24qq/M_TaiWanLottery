"""Read the answer only AFTER sealed predictions exist; never call a predictor."""
import json
import math
from pathlib import Path
import numpy as np
from .protocol import digest,save_new
from ..backtesting import metrics


def score_saved(directory,answer_path):
    directory=Path(directory)
    prediction_path=directory/'prediction.json'
    prediction=json.loads(prediction_path.read_text())
    seal=json.loads((directory/'prediction.seal.json').read_text())
    if digest(prediction_path)!=seal['sha256']:raise ValueError('Prediction was modified')
    input_path=directory/'blind_input.json'
    if digest(input_path)!=prediction['input_sha256']:raise ValueError('Input was modified')
    request=json.loads(input_path.read_text())
    unknown={(i//request['cols']+1,i%request['cols']+1) for i,v in enumerate(request['cells']) if v is None}
    if prediction['target']!=request['target'] or prediction['unknown_cells']!=len(unknown):
        raise ValueError('Prediction does not match request')
    for method in prediction['methods'].values():
        if method['status']!='ready':continue
        ranking=method['ranking']
        positions=[(p['row'],p['col']) for p in ranking]
        if len(positions)!=len(unknown) or set(positions)!=unknown:
            raise ValueError('Ranking must include every unknown cell exactly once')
        if any(not math.isfinite(p['score']) or p['score']<0 for p in ranking):
            raise ValueError('Invalid ranking scores')
    # This is the first answer read. No scoring result is sent to inference.
    answer=json.loads(Path(answer_path).read_text())
    if (answer['rows'],answer['cols'])!=(request['rows'],request['cols']):
        raise ValueError('Answer dimensions mismatch')
    if any(v is not None and v!=answer['cells'][i] for i,v in enumerate(request['cells'])) and not answer.get('visible_shuffle'):
        raise ValueError('Revealed evidence disagrees with answer')
    index=answer['cells'].index(request['target'])
    actual=(index//request['cols']+1,index%request['cols']+1)
    if actual not in unknown:raise ValueError('Target leaked into revealed cells')
    report={name:metrics(method['ranking'],actual) if method['status']=='ready' else
            {'status':'not_ready','reason':method.get('reason')}
            for name,method in prediction['methods'].items()}
    n=len(unknown)
    random={f'top{k}':min(k,n)/n for k in (1,3,5,10)}
    random.update(mrr=sum(1/i for i in range(1,n+1))/n,mean_rank=(n+1)/2)
    result={'physical_board_id':answer['physical_board_id'],'target':request['target'],
            'truth':{'row':actual[0],'col':actual[1]},'unknown_count':n,
            'reveal_ratio':sum(v is not None for v in request['cells'])/len(request['cells']),
            'seed':answer['seed'],'arm':answer['arm'],'protocol':answer['protocol'],
            'prediction_sha256':seal['sha256'],'metrics':report,'random':random}
    save_new(directory/'score.json',result)
    if digest(prediction_path)!=seal['sha256']:raise ValueError('Prediction mutated during scoring')
    return result


def summarize(scores,bootstrap_samples=1000):
    methods=sorted({name for row in scores for name in row['metrics']})
    result={}
    for name in ['random',*methods]:
        rows=[r for r in scores if name=='random' or 'actual_rank' in r['metrics'][name]]
        if not rows:
            result[name]={'status':'not_ready','samples':0};continue
        def value(row,key):return row['random'][key] if name=='random' else row['metrics'][name][key]
        out={'status':'scored','samples':len(rows),'board_count':len({r['physical_board_id'] for r in rows})}
        for key in ('top1','top3','top5','top10','mrr'):
            out[key]=float(np.mean([value(r,key) for r in rows]))
            baseline=float(np.mean([r['random'][key] for r in rows]))
            out[key+'_random']=baseline;out[key+'_lift']=out[key]/baseline
        if name!='random':
            ranks=[r['metrics'][name]['actual_rank'] for r in rows]
            out.update(mean_rank=float(np.mean(ranks)),median_rank=float(np.median(ranks)))
        else:out.update(mean_rank=float(np.mean([r['random']['mean_rank'] for r in rows])),median_rank=None)
        groups=sorted({r['physical_board_id'] for r in rows})
        rng=np.random.default_rng(73)
        cis={}
        for key in ('top1','top3','top5','top10','mrr'):
            by_group={g:[value(r,key)-r['random'][key] for r in rows if r['physical_board_id']==g] for g in groups}
            samples=[]
            for _ in range(bootstrap_samples):
                chosen=rng.choice(groups,len(groups),replace=True)
                samples.append(float(np.mean([v for g in chosen for v in by_group[g]])))
            cis[key]=[float(v) for v in np.quantile(samples,[.025,.975])]
        out['paired_gain_vs_random_board_bootstrap_95ci']=cis
        gain=out['top1']-out['top1_random']
        if out['board_count']<2:label='INSUFFICIENT BOARDS — NO EVIDENCE ABOVE RANDOM'
        elif cis['top1'][0]>0 and out['board_count']>=20:label='EVIDENCE ABOVE RANDOM'
        elif gain>0:label='POSSIBLY ABOVE RANDOM — INSUFFICIENT EVIDENCE'
        elif cis['top1'][1]<0:label='BELOW RANDOM'
        else:label='NO EVIDENCE ABOVE RANDOM'
        out['assessment']=label if name!='random' else 'RANDOM REFERENCE'
        result[name]=out
    return result


def paired_gains(scores):
    keys=('top1','top3','top5','mrr')
    discussion={k:float(np.mean([r['metrics']['discussion'][k]-r['metrics']['round1'][k] for r in scores])) if scores else None for k in keys}
    groups=sorted({r['physical_board_id'] for r in scores})
    cis={}
    rng=np.random.default_rng(91)
    if groups:
        for key in keys:
            by_group={g:[r['metrics']['discussion'][key]-r['metrics']['round1'][key] for r in scores if r['physical_board_id']==g] for g in groups}
            values=[np.mean([v for g in rng.choice(groups,len(groups),replace=True) for v in by_group[g]]) for _ in range(1000)]
            cis[key]=np.quantile(values,[.025,.975]).tolist()
    corrected=sum(r['metrics']['assignment']['top1']>r['metrics']['final_consensus']['top1'] for r in scores)
    damaged=sum(r['metrics']['assignment']['top1']<r['metrics']['final_consensus']['top1'] for r in scores)
    return {'discussion_gain':discussion,'discussion_board_bootstrap_95ci':cis,
            'discussion_effectiveness':'established_on_this_holdout' if len(groups)>=20 and cis.get('top1',[0])[0]>0 else 'discussion effectiveness not established',
            'assignment':{'corrected_wrong_to_right':corrected,'changed_right_to_wrong':damaged,
                          'net_gain':corrected-damaged,'force_assignment_recommended':False,
                          'policy':'No runtime policy change from test outcomes; net_gain <= 0 must not force assignment.'}}
