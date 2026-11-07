"""Sanitized predictor daemon forks one child per trial; never sees test answers."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from .worker import install_read_guard,predict_request
from .. import predictor,models as model_module
from ..registry import train_models
from ..schema import PredictionRequest


def feature_cache(original):
    key=None;template=None
    def features(board,number):
        nonlocal key,template
        current=(board.rows,board.cols,tuple(board.cells))
        if current!=key:
            template=original(board,number);key=current
        result=template.copy();result[:,0]=number/len(board.cells)
        return result
    return features


def prepare(training):
    # Same original fitting code and fixed hyperparameters, fitted once because
    # every trial uses the exact same immutable train history/default policies.
    board=PredictionRequest(rows=10,cols=16,cells=[None]*160,target=73)
    fitted=train_models(board,training['history'])
    if not all(m.ready for m in fitted):raise ValueError('All eight models must be ready')
    history=tuple(tuple(h) for h in training['history'])
    def frozen_models(board,records):
        if (board.rows,board.cols)!=(10,16) or tuple(map(tuple,records))!=history:
            raise ValueError('Frozen training corpus/dimensions mismatch')
        return fitted
    predictor.train_models=frozen_models
    model_module.candidate_features=feature_cache(model_module.candidate_features)
    return fitted


def main(training_path):
    install_read_guard([training_path],Path(training_path).parent)
    training=json.loads(Path(training_path).read_text())
    prepare(training)
    print(json.dumps({'ready':True,'models':8}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        child_output,child_write=os.pipe();ack_read,ack_write=os.pipe()
        pid=os.fork()
        if pid==0:
            os.close(child_output);os.close(ack_write)
            try:
                result=predict_request(request,training)
                result.pop('trace',None)  # Preserve all rankings, omit huge full-board matrices.
                payload=json.dumps({'prediction':result},allow_nan=False).encode()+b'\n'
                with os.fdopen(child_write,'wb') as stream:stream.write(payload);stream.flush()
                # Parent persists prediction + seal before granting this ACK.
                if os.read(ack_read,1)!=b'1':os._exit(2)
                os._exit(0)
            except BaseException as exc:
                try:os.write(child_write,json.dumps({'error':repr(exc)}).encode()+b'\n')
                finally:os._exit(3)
        os.close(child_write);os.close(ack_read)
        with os.fdopen(child_output,'r') as stream:reply=stream.readline()
        print(reply.strip(),flush=True)
        acknowledgement=sys.stdin.readline()
        if acknowledgement.strip()!='SEALED':raise RuntimeError('Missing seal acknowledgement')
        os.write(ack_write,b'1');os.close(ack_write)
        _,status=os.waitpid(pid,0)
        print(json.dumps({'predictor_exited':True,'pid':pid,'exitcode':os.waitstatus_to_exitcode(status)}),flush=True)


if __name__=='__main__':main(sys.argv[1])
