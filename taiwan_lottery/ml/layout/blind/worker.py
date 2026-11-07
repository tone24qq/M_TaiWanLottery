"""Predict-only subprocess: strict request, train-only history, no scoring import."""
import json
import os
from pathlib import Path
import sys

from ..schema import PredictionRequest
from ..predictor import predict


def install_read_guard(allowed_files,scratch=None):
    """Deny answer, corpus, database and unrelated file reads inside the worker.

    Defense in depth for trusted local models, not a hostile-code OS sandbox.
    Imports remain permitted from the runtime and this package only.
    """
    exact={Path(p).resolve() for p in allowed_files}
    scratch=Path(scratch).resolve() if scratch else None
    roots=[Path(sys.base_prefix).resolve()/"lib",Path(sys.prefix).resolve()/"lib",
           Path(__file__).resolve().parents[1],Path('/usr/lib'),Path('/lib')]
    def guard(event,args):
        if event=='open':
            name,mode,flags=args
            if isinstance(name,int):return
            p=Path(os.fsdecode(name)).resolve()
            writing=bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and (scratch is None or not p.is_relative_to(scratch) or p in exact):
                raise PermissionError('blind predictor cannot write outside ephemeral cache')
            if p not in exact and not (scratch and p.is_relative_to(scratch)) and not any(p.is_relative_to(root) for root in roots):
                raise PermissionError('blind predictor denied non-input file read')
        elif event in ('socket.connect','subprocess.Popen','os.system'):
            raise PermissionError('blind predictor cannot access network or subprocesses')
    sys.addaudithook(guard)


def predict_request(request,training):
    if set(request)!={'rows','cols','cells','target'}:
        raise ValueError('Only the four public request fields are allowed')
    if set(training)!={'history','evidence','policy_records'}:
        raise ValueError('Invalid train-only bundle')
    board=PredictionRequest(**request)
    if board.target in board.cells:raise ValueError('Target must be hidden')
    result=predict(board,training['history'],training['evidence'],training['policy_records'])
    methods={p['model']:{'status':p['status'],'ranking':p.get('ranking',[]),
                        'reason':p.get('reason')} for p in result['round1']}
    methods.update({
        'round1':{'status':'ready','ranking':result['round1_consensus']},
        'discussion':{'status':'ready','ranking':result['consensus']['ranking']},
        'final_consensus':{'status':'ready','ranking':result['consensus']['ranking']},
    })
    # Hungarian provides one assigned location, not an independent score ranking.
    # Preserve all other cells in pre-assignment order to define rank 2..N.
    assigned=result['consensus']['target_assignment']
    final=result['consensus']['ranking']
    position=(assigned['row'],assigned['col'])
    methods['assignment']={'status':'ready','ranking':[
        next(p for p in final if (p['row'],p['col'])==position),
        *[p for p in final if (p['row'],p['col'])!=position]],
        'ranking_semantics':'assigned_cell_first_then_unchanged_final_ranking'}
    return {'target':board.target,'unknown_cells':len(board.unknown),
            'methods':methods,'trace':result}


if __name__=='__main__':
    request_path,training_path=sys.argv[1:]
    install_read_guard([request_path,training_path],Path(request_path).parent)
    request=json.loads(Path(request_path).read_text())
    training=json.loads(Path(training_path).read_text())
    print(json.dumps(predict_request(request,training),allow_nan=False))
