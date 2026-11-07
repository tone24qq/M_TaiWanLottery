"""Pre-registered manufacturing baseline: 1000 development + 1000 fresh boards."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import numpy as np
from .protocol import digest,save_new
from .scorer import score_saved
from .calc import frozen_values
from .run import source_manifest

METHODS=['statistical','position','association','lightgbm','catboost','similarity_knn','neural','transformer','round1','discussion','final_consensus','assignment']
MASKS=10


def manufacturing_request(cells):
    """Independent mask entropy only. NEVER generates manufacturing values."""
    if sorted(cells)!=list(range(1,161)):raise ValueError('Invalid manufacturing permutation')
    target_position=cells.index(73)
    visible=secrets.SystemRandom().sample([i for i in range(160) if i!=target_position],80)
    result=[None]*160
    for i in visible:result[i]=cells[i]
    return {'rows':10,'cols':16,'cells':result,'target':73}


def code_hashes():
    root=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*.py'))}


class PredictorDaemon:
    def __init__(self,training):
        self.temp=tempfile.TemporaryDirectory(prefix='manufacturing-predict-only-')
        scratch=Path(self.temp.name);path=scratch/'train.json';path.write_text(json.dumps(training));path.chmod(0o444)
        self.training_sha=hashlib.sha256(json.dumps(training,sort_keys=True).encode()).hexdigest()
        repo=Path(__file__).resolve().parents[4]
        env={k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL')}
        env.update(PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            TMPDIR=str(scratch),TORCHINDUCTOR_CACHE_DIR=str(scratch/'torch-cache'))
        self.errors=(scratch/'stderr.log').open('w')
        self.process=subprocess.Popen([sys.executable,'-m','taiwan_lottery.ml.layout.blind.manufacturing_worker',str(path)],
            cwd=scratch,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.errors,text=True)
        if json.loads(self.line())!={'ready':True,'models':8}:raise RuntimeError('Predictor startup failed')
    def line(self):
        value=self.process.stdout.readline()
        if not value:raise RuntimeError('Predictor stopped: '+(Path(self.temp.name)/'stderr.log').read_text())
        return value
    def predict_seal_exit(self,request,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        save_new(directory/'blind_input.json',request)
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        reply=json.loads(self.line())
        if 'error' in reply:raise RuntimeError(reply['error'])
        prediction=reply['prediction']
        if set(prediction['methods'])!=set(METHODS):raise ValueError('Missing model ranking')
        if prediction['unknown_cells']!=80:raise ValueError('Wrong hidden count')
        if any(m['status']!='ready' or len(m['ranking'])!=80 for m in prediction['methods'].values()):
            raise ValueError('Model not_ready or incomplete ranking')
        prediction.update(input_sha256=digest(directory/'blind_input.json'),training_sha256=self.training_sha)
        sha=save_new(directory/'prediction.json',prediction)
        save_new(directory/'prediction.seal.json',{'sha256':sha})
        # Child is still alive, waiting for this acknowledgement. Only AFTER
        # prediction+seal persistence can it exit; waitpid proof precedes scoring.
        self.process.stdin.write('SEALED\n');self.process.stdin.flush()
        proof=json.loads(self.line())
        if proof.get('predictor_exited') is not True or proof.get('exitcode')!=0:
            raise RuntimeError('Predictor did not exit cleanly')
        save_new(directory/'predictor_exit.json',proof)
        return prediction
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close();self.process.wait(timeout=30)
        self.errors.close();self.temp.cleanup()


def score_excel(directory,snapshot,metadata):
    directory=Path(directory)
    proof=json.loads((directory/'predictor_exit.json').read_text())
    if proof['exitcode']!=0 or not proof['predictor_exited']:raise ValueError('Score before predictor exit')
    seal=json.loads((directory/'prediction.seal.json').read_text())
    if digest(directory/'prediction.json')!=seal['sha256']:raise ValueError('Prediction mutated')
    # FIRST scorer-side read of complete Excel answer occurs here, after both
    # the seal and actual child process termination have been verified.
    if metadata.get('snapshot_sha256') and digest(snapshot)!=metadata['snapshot_sha256']:
        raise ValueError('Frozen Excel snapshot mutated')
    values,_=frozen_values(snapshot)
    save_new(directory/'answer.json',{'rows':10,'cols':16,'cells':values,**metadata})
    return score_saved(directory,directory/'answer.json')


def aggregate(scores,bootstrap=2000):
    ids=sorted({s['physical_board_id'] for s in scores});index={g:i for i,g in enumerate(ids)}
    if not ids:return {}
    group=np.array([index[s['physical_board_id']] for s in scores]);sizes=np.bincount(group)
    rng=np.random.default_rng(20261107)
    weights=np.stack([np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids)) for _ in range(bootstrap)]).astype(float)
    denominators=weights@sizes
    report={}
    for name in METHODS:
        ranks=np.array([s['metrics'][name]['actual_rank'] for s in scores])
        unknown=np.array([s['unknown_count'] for s in scores])
        row={'samples':len(scores),'boards':len(ids),'mean_rank':float(ranks.mean()),'median_rank':float(np.median(ranks))}
        for key,values,random in [*(('top'+str(k),(ranks<=k).astype(float),np.minimum(k,unknown)/unknown) for k in (1,3,5,10)),
                                 ('mrr',1/ranks,np.array([sum(1/j for j in range(1,n+1))/n for n in unknown])),
                                 ('mean_rank',ranks,(unknown+1)/2)]:
            sums=np.bincount(group,weights=values,minlength=len(ids));resampled=(weights@sums)/denominators
            estimate=float(values.mean());baseline=float(random.mean())
            row[key]=estimate;row[key+'_random']=baseline;row[key+'_lift']=estimate/baseline
            row[key+'_95ci']=np.quantile(resampled,[.025,.975]).tolist()
            if key.startswith('top'):row[key+'_hits']=int(values.sum())
            if key=='top1':
                row['top1_familywise_ci']=np.quantile(resampled,[.025/len(METHODS),1-.025/len(METHODS)]).tolist()
                row['evidence_above_random']=row['top1_familywise_ci'][0]>baseline
        # Board bootstrap for median rank via histogram counts (all trials retain
        # their group dependence, not a binomial CI treating masks as boards).
        hist=np.zeros((len(ids),80));np.add.at(hist,(group,ranks-1),1)
        cumulative=(weights@hist).cumsum(axis=1);mid=denominators/2
        left=(cumulative>=mid[:,None]).argmax(axis=1)+1
        right=(cumulative>mid[:,None]).argmax(axis=1)+1
        row['median_rank_95ci']=np.quantile((left+right)/2,[.025,.975]).tolist()
        row['leakage_alert']=row['top1']>=.10
        row['assessment']='SUSPECT LEAKAGE — AUDIT BEFORE ANY CAPABILITY CLAIM' if row['leakage_alert'] else (
            'POSITIVE FAMILYWISE-CORRECTED SIGNAL; REQUIRES FRESH REPLICATION AND AUDIT' if row['evidence_above_random'] else 'NO EVIDENCE ABOVE RANDOM')
        report[name]=row
    return report


def execute_phase(boards,training,output,registry,locked,workers=4):
    if code_hashes()!=locked:raise ValueError('Program changed after lock')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    folders=sorted(Path(boards).glob('board-*'))
    manifest=json.loads((Path(boards)/'calc_manifest.json').read_text())
    if len(folders)!=1000 or manifest['completion_callbacks_verified']!=1000:raise ValueError('Require 1000 confirmed Calc boards')
    # Only preparer sees full frozen values; predictor daemons are independent
    # exec-created processes, never forked from this answer-bearing coordinator.
    for folder in folders:
        values=json.loads((folder/'values.json').read_text())['cells'];signature=tuple(values)
        if signature in registry:raise ValueError('Train/test or phase permutation overlap')
        registry.add(signature)
    totals=[]
    def batch(worker):
        daemon=PredictorDaemon(training);scores=[]
        try:
            for board_index in range(worker,len(folders),workers):
                folder=folders[board_index]
                values=json.loads((folder/'values.json').read_text())['cells']
                snapshot=next(folder.glob('*.xlsx'));masks=set()
                for mask in range(MASKS):
                    request=manufacturing_request(values)
                    pattern=tuple(v is not None for v in request['cells'])
                    if pattern in masks:raise ValueError('Repeated board mask')
                    masks.add(pattern)
                    directory=output/'trials'/f'{board_index:04d}-{mask:02d}'
                    daemon.predict_seal_exit(request,directory)
                    score=score_excel(directory,snapshot,{'physical_board_id':folder.name,
                        'seed':mask,'arm':'baseline','protocol':output.name,
                        'snapshot_sha256':json.loads((folder/'values.json').read_text())['snapshot_sha256']})
                    scores.append(score)
                if len(scores)%250==0:print(f'{output.name}: worker {worker}, {len(scores)} sealed/scored trials',flush=True)
        finally:daemon.close()
        return scores
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(batch,range(workers)):totals.extend(rows)
    if code_hashes()!=locked:raise ValueError('Program changed during holdout')
    if len(totals)!=10000 or len({s['physical_board_id'] for s in totals})!=1000:raise ValueError('Incomplete holdout')
    summary={'phase':output.name,'boards':1000,'trials':10000,'masks_per_board':MASKS,'target':73,'rows':10,'cols':16,
        'visible_count':80,'unknown_count':80,'overall':aggregate(totals),
        'discussion_gain':{k:float(np.mean([s['metrics']['discussion'][k]-s['metrics']['round1'][k] for s in totals])) for k in ('top1','top3','top5','mrr')},
        'discussion_effectiveness':'not established; peer influence not enabled without prior validation evidence',
        'assignment':{'corrected_wrong_to_right':sum(s['metrics']['assignment']['top1']>s['metrics']['final_consensus']['top1'] for s in totals),
                      'changed_right_to_wrong':sum(s['metrics']['assignment']['top1']<s['metrics']['final_consensus']['top1'] for s in totals)},
        'audit':{'sealed_and_child_exited_before_excel_scoring':len(totals),'all_unknown_counts_80':all(s['unknown_count']==80 for s in totals),
            'program_hashes_unchanged':True,'train_test_and_phase_permutations_disjoint':True,
            'mask_entropy':'Fresh OS entropy per trial, independent of Calc; no shared per-board seed/mask template sent to predictor',
            'predictor_input_fields':['rows','cols','cells','target'],'predictor_receives_rng_state_or_hidden_values':False},
        'ci':'2000 resamples of manufacturing boards, retaining all 10 within-board masks; percentile 95% CI, additional familywise Top1 CI across 12 methods',
        'caveat':manifest['caveat']}
    summary['assignment']['net_gain']=summary['assignment']['corrected_wrong_to_right']-summary['assignment']['changed_right_to_wrong']
    save_new(output/'report.json',summary);save_new(output/'scores.json',totals)
    return summary


def main(setup,development,source,program,output,cache_proof):
    setup=Path(setup);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    setup_files=sorted(setup.glob('board-*/values.json'))
    if len(setup_files)!=45:raise ValueError('Require 40 train + 5 integration-only boards')
    source_sha=digest(source)
    for corpus in (setup,Path(development)):
        if json.loads((corpus/'calc_manifest.json').read_text())['source_audit']['workbook_sha256']!=source_sha:
            raise ValueError('Manufacturing source changed across corpora')
    registry={tuple(json.loads(f.read_text())['cells']) for f in setup_files}
    training={'history':[json.loads(f.read_text())['cells'] for f in setup_files[:40]],'evidence':[],'policy_records':[]}
    training_sha=hashlib.sha256(json.dumps(training,sort_keys=True).encode()).hexdigest()
    proof=json.loads(Path(cache_proof).read_text())
    if not proof.get('all_rankings_and_scores_exactly_equal') or proof.get('boards_compared')!=5 or proof.get('training_sha256')!=training_sha:
        raise ValueError('Missing fixed-cache equivalence proof')
    if proof.get('locked_code_hashes')!=code_hashes():raise ValueError('Cache proof does not match current inference sources')
    locked=code_hashes()
    plan={'baseline_work_head':'833a0920b854c430259248cb9fb973255f24d3f8','workbook_sha256':source_sha,**source_manifest(),
        'locked_code_hashes':locked,'training_sha256':training_sha,'training_boards':40,'integration_boards':5,
        'target':73,'rows':10,'cols':16,'visible':80,'hidden':80,'masks_per_board':MASKS,
        'development_boards':1000,'fresh_boards':1000,'total_planned_prediction_trials':20000,
        'model_fit_policy':'Unchanged original train_models/hyperparameters; fit fixed training corpus once, copy to trial children',
        'evidence_policy':'No validation weighting or peer influence; evidence and policy_records remain empty in both phases',
        'feature_cache':'Exact repeated candidate-feature cache only; all five integration boards match original cold predictor exactly',
        'fresh_policy':'Generate new 1000 Calc boards only AFTER development predictions/scoring; no tuning; source hash lock enforced',
        'stats_policy':'2000 board-group bootstrap resamples; 95% CIs, Bonferroni familywise Top1 across 12 methods; Top1 >=10% triggers leakage alert'}
    save_new(output/'protocol_lock.json',plan)
    dev=execute_phase(development,training,output/'development',registry,locked)
    if code_hashes()!=locked:raise ValueError('Sources changed before fresh generation')
    save_new(output/'fresh_generation_started.json',{'development_trials_sealed':10000,'locked_code_hashes':locked})
    env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(Path(program).resolve()),SAL_USE_VCLPLUGIN='svp')
    # New Calc process and a new profile/RNG state. No seed is supplied. No
    # holdout board existed during integration/debug or development scoring.
    subprocess.run([sys.executable,'-m','taiwan_lottery.ml.layout.blind.calc','--source',str(source),
        '--output',str(output/'fresh-boards'),'--count','1000','--program',str(program)],env=env,check=True)
    if code_hashes()!=locked:raise ValueError('Sources changed during fresh generation')
    fresh=execute_phase(output/'fresh-boards',training,output/'fresh',registry,locked)
    result={'development':dev,'fresh':fresh,'total_distinct_manufacturing_boards':len(registry),
        'total_prediction_trials':20000,'runtime_models_or_weights_changed':False,'test_driven_tuning':False,
        'cache_equivalence':proof,'final_result_source':'fresh only',
        'conclusion':'Interpret fresh familywise-corrected, board-group CIs versus conditional random. Large excess is an audit alert, not proof of predictive ability.',
        'microsoft_excel_native_verified':False}
    save_new(output/'manufacturing_baseline.json',result)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('setup','development','source','program','output','cache-proof'):p.add_argument('--'+name,required=True)
    a=p.parse_args();main(a.setup,a.development,a.source,a.program,a.output,a.cache_proof)
