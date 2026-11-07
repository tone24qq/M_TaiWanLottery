"""Compare five sanitized engineering requests with the unchanged cold predictor.

Uses only existing Calc snapshots. Does not generate manufacturing permutations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def check(directory):
    from taiwan_lottery.ml.layout.blind.worker import predict_request,install_read_guard
    from taiwan_lottery.ml.layout.blind.manufacturing_worker import prepare
    from taiwan_lottery.ml.layout.blind.manufacturing import code_hashes
    p=Path(directory).resolve()
    install_read_guard([p/'training.json',p/'requests.json'],p)
    training=json.loads((p/'training.json').read_text());requests=json.loads((p/'requests.json').read_text())
    if len(requests)!=5:raise ValueError('Require five sanitized engineering requests')
    original=[]
    for request in requests:
        result=predict_request(request,training);result.pop('trace');original.append(result)
    prepare(training)
    for index,request in enumerate(requests):
        result=predict_request(request,training);result.pop('trace')
        if result!=original[index]:raise ValueError('Cached predictor changed scores or ranking')
    proof={'boards_compared':5,'all_rankings_and_scores_exactly_equal':True,
        'training_sha256':hashlib.sha256(json.dumps(training,sort_keys=True).encode()).hexdigest(),
        'locked_code_hashes':code_hashes(),
        'description':'All 12 full rankings/scores equal to original cold predictor on five sanitized engineering requests; no answer input.'}
    (p/'proof.json').write_text(json.dumps(proof,indent=2))
    print('Five-request exact equivalence passed.',flush=True)


def main(setup,output):
    from taiwan_lottery.ml.layout.blind.manufacturing import manufacturing_request
    p=Path(output).resolve();p.mkdir(parents=True,exist_ok=False)
    files=sorted(Path(setup).glob('board-*/values.json'))
    if len(files)!=45:raise ValueError('Require 40 train + five engineering snapshots')
    (p/'training.json').write_text(json.dumps({'history':[json.loads(f.read_text())['cells'] for f in files[:40]],'evidence':[],'policy_records':[]}))
    (p/'requests.json').write_text(json.dumps([manufacturing_request(json.loads(f.read_text())['cells']) for f in files[40:]]))
    env={k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL')}
    env.update(PYTHONPATH=str(REPO),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(p),TORCHINDUCTOR_CACHE_DIR=str(p/'torch-cache'))
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(p)],env=env,cwd=p,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup');parser.add_argument('--output');parser.add_argument('--worker')
    args=parser.parse_args()
    if args.worker:check(args.worker)
    else:
        if not args.setup or not args.output:parser.error('--setup and --output required')
        main(args.setup,args.output)
