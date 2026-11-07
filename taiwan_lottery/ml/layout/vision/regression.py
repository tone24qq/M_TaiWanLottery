"""External-fixture regression. Images/labels never enter training or runtime policy."""
from pathlib import Path
import csv
import json
import time
import shutil
import cv2
import numpy as np
from .benchmark import edit_distance


def validate_truth(record):
    rows,cols=record["rows"],record["cols"]
    if not 1<=rows*cols<=400 or rows<1 or cols<1:
        raise ValueError("Invalid ground-truth dimensions")
    cells=record["cells"]
    if len(cells)!=rows*cols:
        raise ValueError("Ground truth must enumerate every coordinate")
    seen=set();values=set()
    for cell in cells:
        coordinate=(cell["row"],cell["col"])
        if not 1<=coordinate[0]<=rows or not 1<=coordinate[1]<=cols or coordinate in seen:
            raise ValueError("Invalid/duplicate truth coordinate")
        seen.add(coordinate)
        state=cell["state"];number=cell["number"]
        if state not in ("OPENED","UNOPENED","UNCERTAIN"):
            raise ValueError("Invalid truth state")
        if state=="OPENED":
            if type(number) is not int or not 1<=number<=rows*cols or number in values:
                raise ValueError("Invalid/duplicate confirmed number")
            values.add(number)
        elif number is not None:
            raise ValueError("Unopened/ambiguous truth must not have a guessed value")
    return True


def recognition_metrics(truth,rankings,values=None,confidences=None):
    n=len(truth);correct=answered=top3=cer=characters=high_error=0
    errors=[]
    for i,(label,ranking) in enumerate(zip(truth,rankings)):
        value=values[i] if values is not None else ranking[0]["value"] if ranking else None
        confidence=confidences[i] if confidences is not None else ranking[0]["score"] if ranking else 0
        hit=value==label
        correct+=hit;answered+=value is not None
        top3+=label in [x["value"] for x in ranking[:3]]
        cer+=edit_distance(str(label),str(value) if value is not None else "")
        characters+=len(str(label));high_error+=not hit and value is not None and confidence>=.8
    return {"samples":n,"correct":correct,"exact_accuracy":correct/n if n else None,
            "top3_recall":top3/n if n else None,"digit_cer":cer/characters if characters else None,
            "coverage":answered/n if n else None,"answered_accuracy":correct/answered if answered else None,
            "abstained_cells":n-answered,"abstention_rate":(n-answered)/n if n else None,
            "wrong_answered_cells":answered-correct,"false_confident_errors":high_error}


def evaluate(record,result):
    validate_truth(record)
    predictions={(p["row"],p["col"]):p for p in result.get("proposals",[])}
    blank={"state":"UNCERTAIN","raw_engines":[],"ensemble":{"candidates":[]},"value":None,"confidence":0}
    opened=[c for c in record["cells"] if c["state"]=="OPENED"]
    ps=[predictions.get((c["row"],c["col"]),blank) for c in opened]
    truth=[c["number"] for c in opened]
    candidates=[p["ensemble"]["candidates"] for p in ps]
    final=recognition_metrics(truth,candidates,[p.get("value") for p in ps],[p.get("confidence",0) for p in ps])
    raw_metrics=recognition_metrics(truth,candidates)
    pre_values=[p.get("unconstrained_value",p["ensemble"]["candidates"][0]["value"]
                    if p["ensemble"]["candidates"] and p["ensemble"]["candidates"][0]["score"]>=.15 else None)
                if p["state"]!="UNOPENED" else None for p in ps]
    pre=recognition_metrics(truth,candidates,pre_values)
    engine_names=sorted({x["engine"] for p in predictions.values() for x in p["raw_engines"]})
    per_engine={}
    for name in engine_names:
        outputs=[next((e for e in p["raw_engines"] if e["engine"]==name),{"candidates":[],"latency_ms":0}) for p in ps]
        if not any(o.get("status") in ("ready","benchmark_only") and o.get("candidates") for o in outputs):
            per_engine[name]={"status":"not_ready"}
            continue
        from .base import numeric
        from .review import native_views
        native_errors=sum(any(v.get("score",0)>=.8 and numeric(v.get("raw_text",""),record["rows"]*record["cols"]) not in (None,label)
                              for v in native_views(o))
                          for label,o in zip(truth,outputs))
        policy=next((e.get("runtime_policy","runtime") for e in result.get("engines",[]) if e["engine"]==name),"runtime")
        per_engine[name]={"status":"ready","runtime_policy":policy,
            **recognition_metrics(truth,[o["candidates"] for o in outputs]),
            "cells_with_wrong_native_confident_view":native_errors,
            "ms_per_called_cell":float(np.mean([o["latency_ms"] for o in outputs if o["latency_ms"]>0]))}
        for label,predicate in [
            ("1-digit",lambda c:len(str(c["number"]))==1),("2-digit",lambda c:len(str(c["number"]))==2),
            ("glare",lambda c:"glare" in c.get("scenarios",[])),("scratch",lambda c:"scratch" in c.get("scenarios",[]))]:
            indices=[i for i,c in enumerate(opened) if predicate(c)]
            per_engine[name][label]=recognition_metrics([truth[i] for i in indices],[outputs[i]["candidates"] for i in indices])
    states={(a,b):0 for a in ("OPENED","UNOPENED","UNCERTAIN") for b in ("OPENED","UNOPENED","UNCERTAIN")}
    unopened_false=[]
    for cell in record["cells"]:
        p=predictions.get((cell["row"],cell["col"]),blank)
        states[(cell["state"],p["state"])]+=1
        if cell["state"]=="UNOPENED" and p.get("value") is not None:
            unopened_false.append({"row":cell["row"],"col":cell["col"],"value":p["value"]})
    state_report={"accuracy":sum(v for (a,b),v in states.items() if a==b)/len(record["cells"]),
                  "confusion":{a:{b:states[(a,b)] for b in ("OPENED","UNOPENED","UNCERTAIN")} for a in ("OPENED","UNOPENED","UNCERTAIN")},
                  "false_opened":sum(v for (a,b),v in states.items() if a=="UNOPENED" and b=="OPENED"),
                  "false_unopened":sum(v for (a,b),v in states.items() if a=="OPENED" and b=="UNOPENED")}
    for state in ("OPENED","UNOPENED"):
        total=sum(v for (a,b),v in states.items() if a==state)
        state_report[state.lower()+"_recall"]=states[(state,state)]/total if total else None
    corners=result.get("corners")
    geometry={"detected":bool(corners)}
    if corners and record.get("corners"):
        actual=np.float32(corners);reference=np.float32(record["corners"])
        intersection,_=cv2.intersectConvexConvex(actual,reference)
        union=cv2.contourArea(actual)+cv2.contourArea(reference)-intersection
        error=float(np.linalg.norm(actual-reference,axis=1).mean())
        cell_size=np.sqrt(cv2.contourArea(reference)/(record["rows"]*record["cols"]))
        geometry.update(iou=float(intersection/union),mean_corner_error_pixels=error,
                        success=bool(intersection/union>.8 and error<cell_size*.25))
    wrong=[]
    for c,p in zip(opened,ps):
        if p.get("value")!=c["number"]:
            reasons=[]
            if not geometry.get("success",True):reasons.append("grid_boundary_misalignment")
            if p["state"]=="UNOPENED":reasons.append("false_unopened")
            if p["state"]=="UNCERTAIN" and not p["raw_engines"]:reasons.append("uncertain_skipped_ocr")
            if p.get("corrected"):reasons.append("constraint_changed_candidate")
            raw={x["engine"]:x.get("candidates",[]) for x in p["raw_engines"]}
            if c["number"] in {x["value"] for values in raw.values() for x in values}:
                reasons.append("truth_present_but_fusion_or_constraint_rejected")
            if p.get("value") is None:reasons.append("abstained")
            else:reasons.append("digit_shape_or_crop_error")
            details={}
            catalog=["tesseract","easyocr","paddle_pp-ocrv6_medium_rec","paddle_pp-ocrv6_small_rec",
                     "paddle_en_pp-ocrv5_mobile_rec","trocr_small","trocr_base",
                     "parseq_tiny","crnn","digit_ctc","digit_segmented"]
            for name in catalog:
                details[name]=next((e for e in p["raw_engines"] if e["engine"]==name),
                    {"status":"not_ready","reason":"not_called_or_not_registered","candidates":[]})
            if p.get("value") is not None and len(str(p["value"]))<len(str(c["number"])):
                reasons.append("possible_missing_leading_digit")
            if c["number"] in {v["value"] for e in p["raw_engines"] if e.get("quality",{}).get("runtime_policy")=="benchmark_only" for v in e["candidates"]}:
                reasons.append("correct_candidate_in_benchmark_only_model")
            wrong.append({"row":c["row"],"col":c["col"],"truth":c["number"],"state":p["state"],
                          "engines":details,"ensemble":p["ensemble"],"final":p.get("value"),
                          "confidence":p.get("confidence",0),"review_reasons":p.get("review_reasons",[]),
                          "failure_reasons":reasons})
    unopened=sum(c["state"]=="UNOPENED" for c in record["cells"])
    return {"image":record["image"],"rows":record["rows"],"cols":record["cols"],"geometry":geometry,
            "final":final,"raw_ensemble":raw_metrics,"before_constraint":pre,"constraint_gain_correct":final["correct"]-pre["correct"],
            "states":state_report,"unopened_false_number_rate":len(unopened_false)/unopened if unopened else 0,
            "unopened_false_numbers":unopened_false,"engines":per_engine,"errors":wrong,
            "performance":result.get("performance"),"engine_inventory":result.get("engines",[])}


def report_saved(fixtures,debug_root,destination):
    records=json.loads(Path(fixtures).read_text())
    metrics=[]
    for record in records:
        path=Path(debug_root)/Path(record["image"]).stem/"result.json"
        evaluated=evaluate(record,json.loads(path.read_text()))
        metrics.append(evaluated)
        (path.parent/"errors.json").write_text(json.dumps(evaluated["errors"],indent=2))
        with (path.parent/"errors.csv").open("w",newline="") as stream:
            writer=csv.writer(stream)
            names=["tesseract","easyocr","paddle_pp-ocrv6_medium_rec","paddle_pp-ocrv6_small_rec",
                   "paddle_en_pp-ocrv5_mobile_rec","trocr_small","trocr_base","parseq_tiny","crnn","digit_ctc","digit_segmented"]
            writer.writerow(["row","col","truth","state",*names,"ensemble","final","confidence","failure_reasons"])
            for error in evaluated["errors"]:
                writer.writerow([error["row"],error["col"],error["truth"],error["state"],
                    *[json.dumps(error["engines"][name].get("candidates",[])) if error["engines"][name].get("candidates") else
                      error["engines"][name].get("status","not_ready") for name in names],
                    json.dumps(error["ensemble"]["candidates"]),error["final"],error["confidence"],";".join(error["failure_reasons"])])
    Path(destination).write_text(json.dumps({"split":"regression_only","images":metrics},indent=2))
    return metrics


def run(fixtures,images_root,output):
    from ..ocr import rectify
    from .service import recognize_board
    from .preprocess import variants,cell_views
    records=json.loads(Path(fixtures).read_text());root=Path(output);root.mkdir(parents=True,exist_ok=True)
    import hashlib
    package=Path(__file__).parent
    (root/"run_manifest.json").write_text(json.dumps({
        "fixture_role":"regression_only","fixture_sha256":hashlib.sha256(Path(fixtures).read_bytes()).hexdigest(),
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob("*.py")},
        "images":len(records)},indent=2))
    for record in records:
        validate_truth(record)
        path=Path(images_root)/record["image"];folder=root/path.stem;folder.mkdir(exist_ok=True)
        shutil.copy2(path,folder/("original"+path.suffix))
        original=path.read_bytes()
        tick=time.perf_counter()
        rectified=rectify(original,record["rows"],record["cols"])
        if isinstance(rectified,dict):result=rectified
        else:
            warped,corners=rectified
            result=recognize_board(original,warped,record["rows"],record["cols"],corners)
            cv2.imwrite(str(folder/"warped.png"),warped)
            overlay=cv2.imread(str(path));cv2.polylines(overlay,[np.int32(corners)],True,(0,0,255),3)
            cv2.imwrite(str(folder/"grid_detection.png"),overlay)
            cell_overlay=warped.copy()
            for row in range(record["rows"]):
                for col in range(record["cols"]):
                    cv2.rectangle(cell_overlay,(col*80,row*80),((col+1)*80-1,(row+1)*80-1),(0,0,255),1)
                    rgb,views,margins=cell_views(warped,row,col)
                    crop=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    location=folder/f"R{row+1}C{col+1}";location.mkdir(exist_ok=True)
                    cv2.imwrite(str(location/"crop.png"),crop)
                    for name,view in views.items():
                        cv2.imwrite(str(location/(name+".png")),cv2.cvtColor(view,cv2.COLOR_RGB2BGR) if view.ndim==3 else view)
            cv2.imwrite(str(folder/"cell_overlay.png"),cell_overlay)
            with (folder/"cells.csv").open("w",newline="") as stream:
                writer=csv.writer(stream);writer.writerow(["row","col","state","raw","ensemble","final","confidence"])
                for p in result["proposals"]:
                    writer.writerow([p["row"],p["col"],p["state"],json.dumps(p["raw_engines"]),p.get("ensemble_value"),p["value"],p["confidence"]])
        result["end_to_end_ms"]=(time.perf_counter()-tick)*1000
        (folder/"result.json").write_text(json.dumps(result,indent=2))
        print(record["image"],result.get("performance",{}),flush=True)
    return report_saved(fixtures,root,root/"metrics.json")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures",required=True)
    parser.add_argument("--images-root",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    run(args.fixtures,args.images_root,args.output)
