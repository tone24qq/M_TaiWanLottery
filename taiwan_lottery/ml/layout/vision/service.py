"""Recognize OPENED cells only; fail engines individually; keep all evidence."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import time
import uuid
import cv2
import numpy as np
from .registry import engines, model_root
from .preprocess import variants, CellStateClassifier
from .consensus import RecognitionConsensus, constrain
from .base import EngineResult


def storage_root():
    return Path(os.getenv("LAYOUT_OCR_DATA_DIR","model_artifacts/layout_digit_dataset")).resolve()


def recognize_board(original,warped,rows,cols,corners,retain=False):
    started=time.perf_counter()
    backends=engines()
    families={b.name:b.family for b in backends}
    calibration_file=model_root()/"calibration.json"
    calibration=json.loads(calibration_file.read_text()) if calibration_file.exists() else {}
    fusion=RecognitionConsensus(calibration)
    profile_file=model_root()/"profile.json"
    profile=json.loads(profile_file.read_text()) if profile_file.exists() else {}
    selected=set(profile.get("engines",[]))
    fingerprints={e.name:e.fingerprint() for e in backends if e.status=="ready"}
    if any(calibration.get("engines",{}).get(name,{}).get("fingerprint")!=fingerprints.get(name) for name in selected):
        selected=set()
    classifier=CellStateClassifier()
    proposals=[]
    capture=str(uuid.uuid4()) if retain else None
    location=storage_root()/"captures"/capture if capture else None
    if location:
        location.mkdir(parents=True)
    for row in range(rows):
        for col in range(cols):
            index=row*cols+col
            bgr=warped[row*80+6:(row+1)*80-6,col*80+6:(col+1)*80-6]
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            state=classifier.classify(rgb)
            outputs=[]
            if state["state"]=="OPENED":
                views=variants(rgb)
                for backend in backends:
                    if selected and backend.name not in selected:
                        continue
                    tick=time.perf_counter()
                    if backend.status!="ready":
                        output=EngineResult(backend.name,status=backend.status,reason=backend.reason).as_dict()
                    else:
                        try:
                            output=backend.recognize(views,rows*cols).as_dict()
                        except Exception as exc:
                            output=EngineResult(backend.name,status="not_ready",
                                reason=type(exc).__name__+": "+str(exc)[:200]).as_dict()
                    output["quality"]["engine_fingerprint"]=backend.fingerprint()
                    output["latency_ms"]=(time.perf_counter()-tick)*1000
                    outputs.append(output)
            ensemble=fusion.fuse([x for x in outputs if not selected or x["engine"] in selected],families,state["quality"])
            p={"row":row+1,"col":col+1,"state":state["state"],"quality":state["quality"],
               "state_method":state["method"],"raw_engines":outputs,"ensemble":ensemble,
               "needs_confirmation":True}
            proposals.append(p)
            if location:
                cv2.imwrite(str(location/f"{index}.png"),bgr)
    before=[p["ensemble"]["candidates"][0]["value"] for p in proposals if p["ensemble"]["candidates"]]
    duplicates=sorted(k for k,v in Counter(before).items() if v>1)
    constrain(proposals,rows*cols)
    elapsed=(time.perf_counter()-started)*1000
    result={"status":"needs_confirmation","rows":rows,"cols":cols,
            "cells":[p["value"] for p in proposals],"proposals":proposals,
            "duplicate_values":duplicates,"corners":corners,
            "capture_id":capture,"image_hash":hashlib.sha256(original).hexdigest(),
            "engines":[b.metadata() for b in backends],
            "performance":{"whole_board_ms":elapsed,
                           "per_cell_ms":elapsed/(rows*cols),"rss_bytes":rss_bytes()},
            "warning":"All values and OPENED/UNOPENED state require human confirmation. Scores are not accuracy."}
    if location:
        (location/"recognition.json").write_text(json.dumps(result,ensure_ascii=False),encoding="utf-8")
    return result


def rss_bytes():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return None
