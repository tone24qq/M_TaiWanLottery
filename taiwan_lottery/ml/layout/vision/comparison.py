"""Descriptive holdout comparisons. Never writes runtime weights or profiles."""
from pathlib import Path
import copy
import itertools
import json
import numpy as np
from .consensus import RecognitionConsensus,constrain
from .regression import evaluate


def compare(fixtures,debug_root,destination):
    truth=json.loads(Path(fixtures).read_text())
    records=[json.loads((Path(debug_root)/Path(t["image"]).stem/"result.json").read_text()) for t in truth]
    names=sorted({e["engine"] for d in records for p in d["proposals"] for e in p["raw_engines"]
                  if e.get("status")=="ready" and e["candidates"]})
    metrics={}
    char_counts=[sum(len(str(c["number"])) for c in t["cells"] if c["state"]=="OPENED") for t in truth]
    if not sum(char_counts):
        report={"status":"not_ready","reason":"no_confirmed_opened_cells","fixture_role":"regression_only"}
        Path(destination).write_text(json.dumps(report))
        return report

    def aggregate(per_image,field="final"):
        n=sum(m[field]["samples"] for m in per_image)
        correct=sum(m[field]["correct"] for m in per_image)
        answered=sum(m[field]["samples"]-m[field]["abstained_cells"] for m in per_image)
        return {"samples":n,"correct":correct,"exact_accuracy":correct/n if n else None,
            "top3_recall":sum(m[field]["top3_recall"]*m[field]["samples"] for m in per_image)/n if n else None,
            "digit_cer":sum(m[field]["digit_cer"]*count for m,count in zip(per_image,char_counts))/sum(char_counts),
            "coverage":answered/n if n else None,"answered_accuracy":correct/answered if answered else None,
            "abstention_rate":1-answered/n if n else None,
            "false_confident_errors":sum(m[field]["false_confident_errors"] for m in per_image)}

    for count in range(1,len(names)+1):
        for subset in itertools.combinations(names,count):
            per_image=[]
            for target,original in zip(truth,records):
                data=copy.deepcopy(original)
                families={e["engine"]:e.get("family",e["engine"]) for e in data.get("engines",[])}
                for p in data["proposals"]:
                    outputs=[e for e in p["raw_engines"] if e["engine"] in subset]
                    p["ensemble"]=RecognitionConsensus().fuse(outputs,families,p["quality"])
                constrain(data["proposals"],target["rows"]*target["cols"])
                report=evaluate(target,data)
                per_image.append(report)
            key="+".join(subset)
            metrics[key]={"engines":list(subset),"before_constraint":aggregate(per_image,"before_constraint"),
                          "after_constraint":aggregate(per_image),
                          "per_image":[{"image":m["image"],"before_constraint":m["before_constraint"],
                                        "after_constraint":m["final"]} for m in per_image]}

    def best(size,constrained):
        field="after_constraint" if constrained else "before_constraint"
        candidates=[(k,m) for k,m in metrics.items() if len(m["engines"])==size]
        if not candidates:return None
        key,method=max(candidates,key=lambda item:(item[1][field]["correct"],
            -item[1][field]["false_confident_errors"],-item[1][field]["digit_cer"]))
        return {"method":key,**method[field]}

    all_candidates=list(metrics.items())
    best_global=max(all_candidates,key=lambda item:(item[1]["after_constraint"]["correct"],
        -item[1]["after_constraint"]["false_confident_errors"])) if all_candidates else None
    multi_candidates=[item for item in all_candidates if len(item[1]["engines"])>=2]
    best_multi=max(multi_candidates,key=lambda item:(item[1]["after_constraint"]["correct"],-item[1]["after_constraint"]["false_confident_errors"])) if multi_candidates else None
    individual={}
    inventory={e["engine"]:e for d in records for e in d.get("engines",[])}
    for name in sorted(inventory):
        indexed=[(i,m["engines"].get(name)) for i,m in enumerate([evaluate(t,d) for t,d in zip(truth,records)])]
        indexed=[(i,m) for i,m in indexed if m and m.get("status")=="ready"]
        available=[m for i,m in indexed]
        if not available:
            individual[name]={"status":"not_ready","inventory":inventory[name]}
            continue
        samples=sum(m["samples"] for m in available)
        individual[name]={"status":"inference_tested","runtime_policy":inventory[name].get("runtime_policy"),
            "samples":samples,"correct":sum(m["correct"] for m in available),
            "exact_accuracy":sum(m["correct"] for m in available)/samples,
            "top3_recall":sum(m["top3_recall"]*m["samples"] for m in available)/samples,
            "digit_cer":sum(m["digit_cer"]*char_counts[i] for i,m in indexed)/sum(char_counts[i] for i,m in indexed),
            "ms_per_cell":sum(m["ms_per_called_cell"]*m["samples"] for m in available)/samples,
            "false_confident_errors":sum(m["false_confident_errors"] for m in available),
            "cells_with_wrong_native_confident_view":sum(m["cells_with_wrong_native_confident_view"] for m in available),
            "subsets":{label:{"samples":sum(m[label]["samples"] for m in available),
                "correct":sum(m[label]["correct"] for m in available)}
                for label in ("1-digit","2-digit","glare","scratch")}}
    report={"fixture_role":"regression_only","individual_engines":individual,"methods":metrics,
            "best_single_on_regression":best(1,False),"best_pair_on_regression":best(2,False),
            "best_triple_on_regression":best(3,False),
            "best_constrained_on_regression":({"method":best_global[0],**best_global[1]["after_constraint"]} if best_global else None),
            "best_multi_engine_constrained_on_regression":({"method":best_multi[0],**best_multi[1]["after_constraint"]} if best_multi else None),
            "validation_selected_runtime_winner":None,
            "runtime_models_promoted_from_regression":[],
            "warning":"Descriptive test-set results only; no model/weight selection or runtime promotion is authorized from these images."}
    Path(destination).write_text(json.dumps(report,indent=2))
    return report
