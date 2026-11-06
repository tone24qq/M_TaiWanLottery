"""Conservative fusion and rectangular Hungarian OCR correction with abstention."""
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


class RecognitionConsensus:
    def __init__(self, calibration=None):
        self.calibration=calibration or {}

    def fuse(self, outputs, families, quality):
        usable=[x for x in outputs if x["status"]=="ready" and x["candidates"]]
        groups=defaultdict(list)
        for output in usable:
            groups[families.get(output["engine"],output["engine"])].append(output)
        if not groups:
            return {"candidates":[],"confidence":0.0,"engines":[],"agreement":False,
                    "weight_mode":"no_evidence","weights":{}}
        weights={}
        supported={}
        for group,members in groups.items():
            for output in members:
                record=self.calibration.get("engines",{}).get(output["engine"],{})
                trusted=(self.calibration.get("kind")=="real_validation" and
                         record.get("samples",0)>=50 and record.get("source_groups",0)>=3 and
                         record.get("fingerprint") is not None and
                         record["fingerprint"]==output.get("quality",{}).get("engine_fingerprint"))
                # Quality subset and length evidence only from validation, never test.
                reliability=1.0
                if trusted:
                    values=[record["exact_accuracy"],record["character_accuracy"],1-record["brier"]]
                    subset="glare" if quality.get("glare",0)>.10 else "blur" if quality.get("blur",100)<35 else "clean"
                    if record.get("subsets",{}).get(subset,{}).get("samples",0)>=10:
                        values.append(record["subsets"][subset]["exact_accuracy"])
                    length_key=str(len(str(output["candidates"][0]["value"])))+"-digit"
                    if record.get("subsets",{}).get(length_key,{}).get("samples",0)>=10:
                        values.append(record["subsets"][length_key]["exact_accuracy"])
                    reliability=max(.001,float(np.mean(values)))
                weights[output["engine"]]=reliability/len(members)/len(groups)
                supported[output["engine"]]=trusted
        total=sum(weights.values())
        weights={k:v/total for k,v in weights.items()}
        values=defaultdict(float)
        for output in usable:
            for candidate in output["candidates"]:
                values[candidate["value"]]+=weights[output["engine"]]*candidate["score"]
        candidates=[{"value":number,"score":float(score)} for number,score in values.items()]
        candidates.sort(key=lambda x:(-x["score"],x["value"]))
        tops={x["candidates"][0]["value"] for x in usable}
        return {"candidates":candidates,"confidence":candidates[0]["score"] if candidates else 0,
                "engines":[x["engine"] for x in usable],"agreement":len(tops)==1,
                "weight_mode":"real_validation" if all(supported.values()) else "equal_family_prior_heuristic",
                "weights":weights,"score_kind":"uncalibrated_evidence_score_not_accuracy"}


def constrain(proposals, maximum):
    opened=[p for p in proposals if p["state"]=="OPENED"]
    # A private dummy for every row lets low/no evidence abstain.
    matrix=np.full((len(opened),maximum+len(opened)),-1.0)
    matrix[:,maximum:]=0.0
    for i,p in enumerate(opened):
        for candidate in p["ensemble"]["candidates"]:
            if 1<=candidate["value"]<=maximum and candidate["score"]>=.15:
                matrix[i,candidate["value"]-1]=candidate["score"]
    if opened:
        rr,cc=linear_sum_assignment(-matrix)
        assigned={int(r):int(c) for r,c in zip(rr,cc)}
    else:
        assigned={}
    for i,p in enumerate(opened):
        c=assigned[i]
        p["value"]=c+1 if c<maximum and matrix[i,c]>0 else None
        p["confidence"]=float(matrix[i,c]) if p["value"] is not None else 0.0
    for p in proposals:
        if p["state"]!="OPENED":
            p["value"],p["confidence"]=None,0.0
        raw=p["ensemble"]["candidates"]
        before=raw[0]["value"] if raw else None
        p["ensemble_value"]=before
        p["corrected"]=before!=p["value"] and p["state"]=="OPENED"
        p["correction_reason"]=("global_unique_number_assignment" if p["value"] is not None else
                                "abstain_no_supported_unique_value") if p["corrected"] else None
        p["disputed"]=(p["state"]=="UNCERTAIN" or (p["state"]=="OPENED" and
                        (p["confidence"]<.65 or not p["ensemble"]["agreement"] or p["corrected"])))
    return proposals
