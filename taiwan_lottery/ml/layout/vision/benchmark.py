"""Real-image benchmark from frozen engine outputs, with disjoint selection/test.

Synthetic samples are explicitly excluded from official validation and reports.
"""
from collections import Counter,defaultdict
import itertools
import numpy as np
from .consensus import RecognitionConsensus, constrain


def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def cell_metrics(truth,predictions):
    if not truth:
        return {"samples":0,"exact_accuracy":None,"digit_cer":None,"top3_recall":None}
    hits=errors=chars=top3=0
    for label,candidates in zip(truth,predictions):
        text=str(label); guess=str(candidates[0]["value"]) if candidates else ""
        hits+=guess==text
        errors+=edit_distance(text,guess);chars+=len(text)
        top3+=label in [x["value"] for x in candidates[:3]]
    return {"samples":len(truth),"exact_accuracy":hits/len(truth),
            "digit_cer":errors/max(1,chars),"top3_recall":top3/len(truth)}


def score(samples,engine_names=None,constrained=False,calibration=None):
    by_board=defaultdict(list)
    for s in samples:
        by_board[s["capture_id"]].append(s)
    truth=[];predictions=[];board_correct=wrong=duplicates=corrections=wrong_corrections=complete_boards=0
    for board in by_board.values():
        positions={(s["raw_outputs"].get("row"),s["raw_outputs"].get("col")) for s in board}
        complete=(len(positions)==board[0]["rows"]*board[0]["cols"] and
                  all(s["state"]=="UNOPENED" or (s["state"]=="OPENED" and s["label"] is not None) for s in board))
        proposals=[]
        for sample in board:
            raw=[x for x in sample["raw_outputs"]["raw_engines"]
                 if (engine_names is None or x["engine"] in engine_names) and
                 (not x["engine"].startswith("digit_") or
                  ("training_sources" in x.get("quality",{}) and
                   sample["source_board"] not in x["quality"]["training_sources"]))]
            families={x["engine"]:x["engine"].split("_")[0] for x in raw}
            fusion=RecognitionConsensus(calibration).fuse(raw,families,sample["quality"])
            proposals.append({"state":sample["state"],"ensemble":fusion,"sample":sample})
        before=[p["ensemble"]["candidates"][0]["value"] for p in proposals if p["ensemble"]["candidates"]]
        duplicates+=sum(n-1 for n in Counter(before).values() if n>1)
        if constrained:
            constrain(proposals,board[0]["rows"]*board[0]["cols"])
        errors=0
        for p in proposals:
            sample=p["sample"]
            if sample["state"]!="OPENED" or sample["label"] is None:
                continue
            candidates=p["ensemble"]["candidates"]
            if constrained:
                if p["corrected"]:
                    corrections+=1
                    wrong_corrections+=int(bool(p["ensemble"]["candidates"]) and
                        p["ensemble"]["candidates"][0]["value"]==sample["label"] and p["value"]!=sample["label"])
                # A globally reassigned first choice must also lead the evaluated ranking.
                candidates=([{"value":p["value"],"score":p["confidence"]}] +
                            [x for x in candidates if x["value"]!=p["value"]]) if p["value"] else []
            truth.append(sample["label"]);predictions.append(candidates)
            errors+=not candidates or candidates[0]["value"]!=sample["label"]
        wrong+=errors
        if complete:
            complete_boards+=1
            board_correct+=errors==0
    return {**cell_metrics(truth,predictions),"boards":len(by_board),
            "complete_board_samples":complete_boards,
            "all_opened_cells_correct_rate":board_correct/complete_boards if complete_boards else None,
            "wrong_cells":wrong,"duplicate_count_before_constraint":duplicates,
            "global_corrections":corrections,"wrong_global_corrections":wrong_corrections}


def benchmark(samples):
    from .dataset import grouped_split
    samples=[s for s in samples if not s.get("synthetic",False) and s.get("split") not in ("regression","regression_only")]
    split=grouped_split(samples)
    if not split["validation"] or not split["test"]:
        return {"status":"not_ready","reason":split.get("reason","missing_real_validation_test"),
                "accuracy_conclusion":"尚無真實照片 accuracy 結論"}
    names=sorted({x["engine"] for s in samples for x in s["raw_outputs"]["raw_engines"]
                  if x["status"]=="ready" and x["candidates"]})
    if not names:
        return {"status":"not_ready","reason":"no_successful_engine_outputs"}
    calibration={"kind":"real_validation","engines":{}}
    for name in names:
        usable=[s for s in split["validation"] if s["state"]=="OPENED" and s["label"] is not None]
        candidates=[next((x["candidates"] for x in s["raw_outputs"]["raw_engines"] if x["engine"]==name),[]) for s in usable]
        m=cell_metrics([s["label"] for s in usable],candidates)
        versions={x.get("quality",{}).get("engine_fingerprint") for s in usable
                  for x in s["raw_outputs"]["raw_engines"] if x["engine"]==name}
        fingerprint=next(iter(versions)) if len(versions)==1 else None
        # Reliability uses held-out validation cells only.
        if usable:
            brier=np.mean([((c[0]["score"] if c else 0) - int(bool(c) and c[0]["value"]==s["label"]))**2
                           for s,c in zip(usable,candidates)])
            calibration["engines"][name]={**m,"fingerprint":fingerprint,"character_accuracy":max(0,1-m["digit_cer"]),
                "brier":float(brier),"source_groups":len({s["source_board"] for s in usable}),"subsets":{}}
            for scenario in ("clean","glare","scratch","blur","perspective","low-light","1-digit","2-digit","3-digit"):
                selected=[i for i,s in enumerate(usable) if scenario in s.get("scenarios",[]) or
                          scenario==f'{len(str(s["label"]))}-digit']
                sm=cell_metrics([usable[i]["label"] for i in selected],[candidates[i] for i in selected])
                calibration["engines"][name]["subsets"][scenario]=sm
    # Compare all nonempty engine subsets from frozen outputs: no extra inference.
    methods={}
    for n in range(1,len(names)+1):
        for subset in itertools.combinations(names,n):
            for constrained in (False,True):
                key="+".join(subset)+(":constraint" if constrained else "")
                methods[key]={"engines":list(subset),"constrained":constrained,
                    "validation":score(split["validation"],subset,constrained,calibration)}
    eligible=[key for key,method in methods.items() if method["constrained"]]
    best=max(eligible,key=lambda k:((methods[k]["validation"]["exact_accuracy"] or 0),
                                   -len(methods[k]["engines"]),not methods[k]["constrained"]))
    # Test set is only touched after weights and best method are frozen.
    report={}
    for key,method in methods.items():
        # Publish singles, full ensemble and the selected method, avoid a huge report.
        if key==best or len(method["engines"]) in (1,len(names)):
            report[key]={**method,"test":score(split["test"],method["engines"],method["constrained"],calibration),
                         "test_scenarios":{scenario:score([s for s in split["test"] if scenario in s.get("scenarios",[])],
                             method["engines"],method["constrained"],calibration)
                             for scenario in ("clean","glare","scratch","blur","perspective","low-light")}}
    return {"status":"ready","selection":"validation_only","best_method":methods[best],
            "methods":report,"calibration":calibration,
            "split_ids":{key:[s.get("id") for s in values] for key,values in split.items()},
            "warning":"Accuracy is conditional on these real source boards; a small test set is not a guarantee"}
