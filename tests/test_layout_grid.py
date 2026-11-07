"""Geometry and safeguards independent of the three private regression photos."""
import cv2
import numpy as np
import pytest
from taiwan_lottery.ml.layout.vision.grid import detect_grid
from taiwan_lottery.ml.layout.vision.preprocess import CellStateClassifier,cell_views
from taiwan_lottery.ml.layout.vision.base import numeric
from taiwan_lottery.ml.layout.vision.engines import fuse_views
from taiwan_lottery.ml.layout.vision.consensus import constrain
from taiwan_lottery.ml.layout.vision.regression import validate_truth


def synthetic_grid(rows,cols):
    board=np.full((rows*56+1,cols*64+1,3),220,np.uint8)
    for r in range(rows):
        for c in range(cols):
            cv2.circle(board,(c*64+32,r*56+28),20,(75,75,75),-1)
            if (r+c)%3:
                cv2.circle(board,(c*64+32,r*56+28),14,(220,220,220),-1)
                cv2.putText(board,str((r*cols+c)%99+1),(c*64+20,r*56+35),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1)
    for r in range(rows+1):cv2.line(board,(0,r*56),(cols*64,r*56),(70,70,70),1)
    for c in range(cols+1):cv2.line(board,(c*64,0),(c*64,rows*56),(70,70,70),1)
    return board


def test_grid_prefers_repeated_lattice_over_larger_decoy_rectangle():
    rows,cols=6,8
    board=synthetic_grid(rows,cols)
    canvas=np.full((720,1100,3),190,np.uint8)
    canvas[320:320+board.shape[0],450:450+board.shape[1]]=board
    cv2.rectangle(canvas,(30,40),(400,250),(60,60,60),3)
    points,debug=detect_grid(canvas,rows,cols)
    assert points is not None,debug
    expected=np.float32([[450,320],[450+cols*64,320],[450+cols*64,320+rows*56],[450,320+rows*56]])
    intersect,_=cv2.intersectConvexConvex(points,expected)
    union=cv2.contourArea(points)+cv2.contourArea(expected)-intersect
    assert intersect/union>.85,(points,debug)


def test_cover_detector_uses_local_contrast_not_absolute_brightness():
    rgb=np.full((80,80,3),170,np.uint8)
    cv2.circle(rgb,(40,40),30,(115,115,115),-1)
    state=CellStateClassifier().classify(rgb)
    assert state["state"]=="UNOPENED"
    assert state["quality"]["circle_fill_ratio"]>.94
    opened=rgb.copy()
    cv2.circle(opened,(40,40),20,(170,170,170),-1)
    cv2.putText(opened,"03",(25,47),cv2.FONT_HERSHEY_SIMPLEX,.5,(55,55,55),1)
    assert CellStateClassifier().classify(opened)["state"]!="UNOPENED"


def test_crop_modes_and_leading_zero_evidence_preserved():
    image=np.full((80,80,3),200,np.uint8)
    cv2.rectangle(image,(1,1),(78,78),(20,20,20),1)
    _,views,margins=cell_views(image,0,0)
    assert {"crop_loose","crop_tight","number_roi","inverted_number_roi"}<=views.keys()
    assert all(4<=m<=10 for m in margins.values())
    result=fuse_views("engine",[("03",.9),("3",.8)],100)
    assert result.candidates[0]["value"]==3 and result.raw_text==["03","3"]
    assert numeric("08",60)==8


def test_stronger_primary_not_displaced_to_fill_weaker_cell():
    def p(c):return {"state":"OPENED","ensemble":{"candidates":[{"value":n,"score":s} for n,s in c],"agreement":True}}
    values=[p([(1,.7),(2,.10)]),p([(1,.5)])]
    constrain(values,2)
    assert values[0]["value"]==1
    assert values[1]["value"] is None


def test_truth_rejects_guessing_uncertain_and_duplicate_values():
    base={"rows":1,"cols":2,"cells":[
        {"row":1,"col":1,"state":"OPENED","number":1},
        {"row":1,"col":2,"state":"UNCERTAIN","number":None}]}
    assert validate_truth(base)
    base["cells"][1]["number"]=2
    with pytest.raises(ValueError):validate_truth(base)
    base["cells"][1].update(state="OPENED",number=1)
    with pytest.raises(ValueError):validate_truth(base)


def test_regression_records_cannot_become_training_or_validation():
    from taiwan_lottery.ml.layout.vision.dataset import grouped_split,stage
    from taiwan_lottery.ml.layout.vision.benchmark import benchmark
    records=[{"id":i,"source_board":str(i//20),"image_hash":str(i//20),
              "label":1,"state":"OPENED","split":"regression_only"} for i in range(600)]
    assert stage(records)=="not_ready"
    split=grouped_split(records)
    assert not split["train"] and not split["validation"] and not split["test"]
    assert benchmark(records)["status"]=="not_ready"


def test_package_presence_alone_does_not_make_recognizer_ready():
    from taiwan_lottery.ml.layout.vision.registry import Unavailable
    engine=Unavailable("test","weights not provided")
    engine.readiness={"package_installed":True,"model_downloaded":False,
        "checksum_verified":False,"license_verified":False,"inference_tested":False}
    assert engine.metadata()["ready"] is False
    assert engine.metadata()["package_installed"] is True
    assert engine.metadata()["model_downloaded"] is False


def test_high_score_does_not_hide_possible_truncated_number():
    from taiwan_lottery.ml.layout.vision.review import apply_review,native_views
    proposal={"state":"OPENED","value":8,"confidence":.95,"corrected":False,
        "ensemble":{"agreement":True,"candidates":[{"value":8,"score":.95},{"value":82,"score":0}]},
        "raw_engines":[],"quality":{}}
    apply_review(proposal,100)
    assert proposal["value"]==8  # Review cannot alter inference or fabricate 82.
    assert "potentially_truncated_number" in proposal["review_reasons"]
    assert proposal["needs_confirmation"]
    output={"quality":{"view_results":[{"view":"roi","raw_text":"82","score":.9},
                                        {"view":"roi","raw_text":"8","score":.85}]}}
    assert list(native_views(output))==[{"view":"roi","raw_text":"82","score":.9}]
