"""Mask generation and immutable prediction persistence. No answer scoring."""
import hashlib
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from ..schema import HistoryRequest


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    path.chmod(0o444)
    return digest(path)


def validate_corpus(records):
    seen_boards=set();groups=set()
    for record in records:
        if record.get('provenance')!='real_confirmed':
            raise ValueError('Only real_confirmed full boards can estimate real performance')
        group=record.get('physical_board_id')
        if not isinstance(group,str) or not group:raise ValueError('physical_board_id required')
        board=HistoryRequest(rows=record['rows'],cols=record['cols'],cells=record['cells'],source=group)
        signature=(board.rows,board.cols,tuple(board.cells))
        if signature in seen_boards or group in groups:
            raise ValueError('Duplicate physical board or complete permutation')
        seen_boards.add(signature);groups.add(group)


def observation_time(record):
    value=datetime.fromisoformat(record["observed_at"].replace("Z","+00:00"))
    if value.tzinfo is None:raise ValueError("observed_at needs an explicit timezone")
    return value.astimezone(timezone.utc)


def partition(records):
    validate_corpus(records)
    # A declared timestamp is required for chronological claims.
    chronological=bool(records) and all(r.get('observed_at') for r in records)
    ordered=sorted(records,key=lambda r:(observation_time(r),r['physical_board_id'])) if chronological else sorted(records,key=lambda r:r['physical_board_id'])
    n=len(ordered)
    if n<5:return {'train':[],'validation':[],'test':[],'unallocated':ordered,'chronological':chronological}
    train=max(1,int(n*.6));validation=max(1,int(n*.2))
    return {'train':ordered[:train],'validation':ordered[train:train+validation],
            'test':ordered[train+validation:],'unallocated':[],'chronological':chronological}


def masked_request(record,target,seed,revealed_fraction=.5,visible_shuffle=False):
    n=record['rows']*record['cols']
    if not 1<=target<=n:raise ValueError('Target outside board')
    if not 0<revealed_fraction<1:raise ValueError('Reveal fraction must be between zero and one')
    rng=np.random.default_rng(np.random.SeedSequence([seed,target,n]))
    target_index=record['cells'].index(target)
    available=[i for i in range(n) if i!=target_index]
    count=min(n-1,int(n*revealed_fraction))
    positions=rng.choice(available,count,replace=False).tolist()
    values=[record['cells'][i] for i in positions]
    if visible_shuffle:rng.shuffle(values)
    cells=[None]*n
    for i,v in zip(positions,values):cells[i]=v
    return {'rows':record['rows'],'cols':record['cols'],'cells':cells,'target':target}


def run_predictor(request,training,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    input_path=directory/'blind_input.json'
    save_new(input_path,request)
    # Worker cwd has only sanitized inputs, never answers, board IDs or paths.
    with tempfile.TemporaryDirectory(prefix='position-blind-') as temp:
        temp=Path(temp)
        req=temp/'input.json';train=temp/'train.json'
        req.write_text(json.dumps(request));train.write_text(json.dumps(training))
        (temp/'cache').mkdir()
        repo=Path(__file__).resolve().parents[4]
        env={k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL','LD_LIBRARY_PATH')}
        env.update(PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(temp/'cache'),TORCHINDUCTOR_CACHE_DIR=str(temp/'cache'/'torch'))
        result=subprocess.run([sys.executable,'-m','taiwan_lottery.ml.layout.blind.worker',str(req),str(train)],
            cwd=temp,env=env,text=True,capture_output=True,check=True,timeout=600)
        prediction=json.loads(result.stdout)
    prediction['input_sha256']=digest(input_path)
    prediction['training_sha256']=hashlib.sha256(json.dumps(training,sort_keys=True).encode()).hexdigest()
    sha=save_new(directory/'prediction.json',prediction)
    save_new(directory/'prediction.seal.json',{'sha256':sha})
    return prediction
