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
        if candidates:
            top=candidates[0]["value"]
            supporters={families.get(x["engine"],x["engine"]) for x in usable if x["candidates"][0]["value"]==top}
            for candidate in candidates:
                if len(supporters)<2 or any(x["candidates"][0]["value"]!=top for x in usable):
                    candidate["score"]=min(candidate["score"],.79)
        tops={x["candidates"][0]["value"] for x in usable}
        return {"candidates":candidates,"confidence":candidates[0]["score"] if candidates else 0,
                "engines":[x["engine"] for x in usable],"agreement":len(tops)==1,
                "weight_mode":"real_validation" if all(supported.values()) else "equal_family_prior_heuristic",
                "weights":weights,"score_kind":"uncalibrated_evidence_score_not_accuracy"}


def constrain(proposals, maximum):
    opened=[p for p in proposals if p["state"] in ("OPENED","UNCERTAIN")]
    # A private dummy for every row lets low/no evidence abstain.
    matrix=np.full((len(opened),maximum+len(opened)),-1.0)
    matrix[:,maximum:]=0.0
    for i,p in enumerate(opened):
        candidates=p["ensemble"]["candidates"]
        best=candidates[0]["score"] if candidates else 0.0
        abstain=max(.15,best*.75)
        matrix[i,maximum:]=abstain
        for candidate in candidates:
            if 1<=candidate["value"]<=maximum and candidate["score"]>=abstain:
                matrix[i,candidate["value"]-1]=candidate["score"]
    # Lexicographic objective: first preserve the strongest primary observation
    # of each value. Global optimization may resolve duplicate losers using
    # supported alternatives, but cannot displace a stronger primary solely to
    # increase aggregate coverage with a weaker row.
    winners={}
    for i,p in enumerate(opened):
        candidates=p["ensemble"]["candidates"]
        if candidates and candidates[0]["score"]>=.15:
            number=candidates[0]["value"]
            margin=candidates[0]["score"]-(candidates[1]["score"] if len(candidates)>1 else 0)
            previous=winners.get(number)
            old_candidates=opened[previous]["ensemble"]["candidates"] if previous is not None else []
            old_margin=(old_candidates[0]["score"]-(old_candidates[1]["score"] if len(old_candidates)>1 else 0)) if old_candidates else -1
            if previous is None or margin>old_margin:
                winners[number]=i
    assigned={row:number-1 for number,row in winners.items()}
    unresolved=[i for i in range(len(opened)) if i not in assigned]
    if unresolved:
        remaining=matrix[unresolved].copy()
        for number in winners:remaining[:,number-1]=-1
        rr,cc=linear_sum_assignment(-remaining)
        assigned.update({unresolved[int(r)]:int(c) for r,c in zip(rr,cc)})
    for i,p in enumerate(opened):
        c=assigned.get(i,maximum)
        p["unconstrained_value"]=(p["ensemble"]["candidates"][0]["value"]
                                  if p["ensemble"]["candidates"] and p["ensemble"]["candidates"][0]["score"]>=.15 else None)
        p["value"]=c+1 if c<maximum and matrix[i,c]>0 else None
        p["confidence"]=min(float(matrix[i,c]),.79) if p["state"]=="UNCERTAIN" and p["value"] is not None else float(matrix[i,c]) if p["value"] is not None else 0.0
    for p in proposals:
        if p["state"]=="UNOPENED":
            p["value"],p["confidence"]=None,0.0
        raw=p["ensemble"]["candidates"]
        before=raw[0]["value"] if raw else None
        p["ensemble_value"]=before
        p["corrected"]=before!=p["value"] and p["state"] in ("OPENED","UNCERTAIN")
        p["correction_reason"]=("global_unique_number_assignment" if p["value"] is not None else
                                "abstain_no_supported_unique_value") if p["corrected"] else None
        p["disputed"]=(p["state"]=="UNCERTAIN" or (p["state"]=="OPENED" and
                        (p["confidence"]<.65 or not p["ensemble"]["agreement"] or p["corrected"])))
    return proposals
