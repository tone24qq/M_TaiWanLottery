"""Opt-in audit of actual Calc-produced, sealed manufacturing trial artifacts."""
import hashlib
import json
import os
from pathlib import Path
import pytest
from taiwan_lottery.ml.layout.blind.manufacturing import METHODS,code_hashes

ROOT=os.getenv('LAYOUT_MANUFACTURING_RUN')
pytestmark=pytest.mark.skipif(not ROOT,reason='private manufacturing run artifacts not supplied')


def test_complete_fresh_manufacturing_holdout_and_seals():
    root=Path(ROOT)
    report=json.loads((root/'manufacturing_baseline.json').read_text())
    lock=json.loads((root/'protocol_lock.json').read_text())
    assert lock['locked_code_hashes']==code_hashes()
    assert report['total_prediction_trials']==20000
    assert report['total_distinct_manufacturing_boards']==2045
    assert not report['runtime_models_or_weights_changed']
    assert not report['test_driven_tuning']
    assert not report['microsoft_excel_native_verified']
    assert (root/'fresh_generation_started.json').stat().st_mtime_ns >= (root/'development/scores.json').stat().st_mtime_ns
    assert (root/'fresh-boards/board-000000/values.json').stat().st_mtime_ns >= (root/'fresh_generation_started.json').stat().st_mtime_ns
    all_boards=set()
    for phase in ('development','fresh'):
        summary=report[phase]
        assert summary['boards']==1000 and summary['trials']==10000
        assert summary['audit']['sealed_and_child_exited_before_excel_scoring']==10000
        assert summary['audit']['all_unknown_counts_80']
        trials=sorted((root/phase/'trials').iterdir())
        assert len(trials)==10000
        boards={}
        for trial in trials:
            request=json.loads((trial/'blind_input.json').read_text())
            assert set(request)=={'rows','cols','cells','target'}
            assert request['target']==73
            unknown={(i//16+1,i%16+1) for i,v in enumerate(request['cells']) if v is None}
            assert len(unknown)==80 and 73 not in request['cells']
            raw=(trial/'prediction.json').read_bytes();prediction=json.loads(raw)
            assert hashlib.sha256(raw).hexdigest()==json.loads((trial/'prediction.seal.json').read_text())['sha256']
            assert prediction['training_sha256']==lock['training_sha256']
            proof=json.loads((trial/'predictor_exit.json').read_text())
            assert proof['predictor_exited'] and proof['exitcode']==0
            times=[(trial/name).stat().st_mtime_ns for name in ('prediction.json','prediction.seal.json','predictor_exit.json','answer.json','score.json')]
            assert times==sorted(times)
            answer=json.loads((trial/'answer.json').read_text())
            assert sorted(answer['cells'])==list(range(1,161))
            index=answer['cells'].index(73);actual=(index//16+1,index%16+1)
            assert actual in unknown
            score=json.loads((trial/'score.json').read_text())
            assert score['unknown_count']==80 and score['random']['top1']==1/80
            assert set(prediction['methods'])==set(METHODS)
            for name,method in prediction['methods'].items():
                ranking=method['ranking'];positions=[(p['row'],p['col']) for p in ranking]
                assert method['status']=='ready' and len(positions)==80 and set(positions)==unknown
                assert score['metrics'][name]['actual_rank']==positions.index(actual)+1
            board_id=score['physical_board_id'];signature=tuple(answer['cells'])
            assert board_id not in boards or boards[board_id]==signature
            boards[board_id]=signature
        assert len(boards)==1000 and len(set(boards.values()))==1000
        assert not (all_boards & set(boards.values()))
        all_boards.update(boards.values())
        for method in summary['overall'].values():
            assert method['samples']==10000 and method['boards']==1000
            for k in (1,3,5,10):assert method[f'top{k}_hits']/10000==method[f'top{k}']
