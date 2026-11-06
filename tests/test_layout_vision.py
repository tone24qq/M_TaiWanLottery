"""Small deterministic checks; synthetic accuracy is not a phone-photo claim."""
import json
from io import BytesIO
import numpy as np
import pytest
from PIL import Image,ImageDraw,ImageFont
from taiwan_lottery.ml.layout.vision.base import numeric
from taiwan_lottery.ml.layout.vision.preprocess import variants,CellStateClassifier
from taiwan_lottery.ml.layout.vision.consensus import RecognitionConsensus,constrain
from taiwan_lottery.ml.layout.vision.engines import fuse_views
from taiwan_lottery.ml.layout.vision.dataset import grouped_split,stage
from taiwan_lottery.ml.layout.vision.benchmark import benchmark,cell_metrics
from taiwan_lottery.ml.layout.vision.digit import ctc_candidates


def cell(text="73"):
    image=Image.new("RGB",(100,70),"white")
    font=ImageFont.truetype("DejaVuSans.ttf",42)
    ImageDraw.Draw(image).text((15,8),text,fill="black",font=font)
    return np.array(image)


def output(engine,candidates):
    return {"engine":engine,"status":"ready","candidates":[{"value":v,"score":s} for v,s in candidates],
            "quality":{}}


def proposal(state,candidates):
    return {"state":state,"ensemble":{"candidates":[{"value":v,"score":s} for v,s in candidates],"agreement":False}}


def test_numeric_and_nine_views():
    assert numeric(" 73 ",100)==73
    assert numeric("73x",100) is None
    assert numeric("101",100) is None
    assert numeric("0",100) is None
    assert len(variants(cell()))==9


def test_state_is_independent_of_ocr():
    classifier=CellStateClassifier()
    assert classifier.classify(np.full((60,60,3),50,np.uint8))["state"]=="UNOPENED"
    assert classifier.classify(cell())["state"]=="OPENED"
    image=np.zeros((60,60,3),np.uint8);image[:30]=255
    assert classifier.classify(image)["state"]=="UNCERTAIN"


def test_one_engine_vote_not_nine_augmentation_votes():
    fused=fuse_views("a",[("73",.8)]*9,100)
    assert fused.candidates[0]["score"]==pytest.approx(.8)
    result=RecognitionConsensus().fuse([fused.as_dict(),output("b",[(13,.9)])],{"a":"a","b":"b"},{})
    assert result["weights"]=={"a":.5,"b":.5}
    # Three Paddle variants still share one family prior.
    result=RecognitionConsensus().fuse([output("p1",[(73,.9)]),output("p2",[(73,.9)]),output("b",[(13,.9)])],
                                      {"p1":"p","p2":"p","b":"b"},{})
    assert result["weights"]["p1"]+result["weights"]["p2"]==pytest.approx(.5)


def test_constraint_corrects_only_opened_supported_values():
    p=[proposal("OPENED",[(13,.48),(73,.45)]),proposal("OPENED",[(13,.97)]),
       proposal("UNOPENED",[(99,.99)]),proposal("UNCERTAIN",[(55,.99)]),proposal("OPENED",[])]
    constrain(p,100)
    assert [x["value"] for x in p]==[73,13,None,None,None]
    assert p[0]["corrected"]
    assert p[2]["corrected"] is False


def test_duplicate_without_supported_alternative_abstains():
    p=[proposal("OPENED",[(13,.9)]),proposal("OPENED",[(13,.8)])]
    constrain(p,100)
    assert [x["value"] for x in p]==[13,None]
    assert all(x["value"] in (13,None) for x in p)


def test_split_groups_boards_and_same_original_image():
    samples=[]
    for b in range(12):
        for c in range(3):
            samples.append({"id":b*3+c,"source_board":str(b),"image_hash":"same" if b in (0,1) else str(b),
                            "label":c+1,"state":"OPENED"})
    split=grouped_split(samples)
    boards=[{s["source_board"] for s in split[k]} for k in ("train","validation","test")]
    assert not boards[0]&boards[1] and not boards[0]&boards[2] and not boards[1]&boards[2]
    assert next(i for i,b in enumerate(boards) if "0" in b)==next(i for i,b in enumerate(boards) if "1" in b)
    assert stage(samples)=="not_ready"


def test_no_fake_real_accuracy_and_metrics():
    assert benchmark([])["accuracy_conclusion"]=="尚無真實照片 accuracy 結論"
    m=cell_metrics([73,13],[[{"value":73,"score":.8}],[{"value":18,"score":.7},{"value":13,"score":.2}]])
    assert m["exact_accuracy"]==.5
    assert m["top3_recall"]==1
    assert m["digit_cer"]==.25


def test_ctc_repeated_digits():
    a=np.full((3,11),1e-8)
    a[0,1]=1;a[1,10]=1;a[2,1]=1
    assert ctc_candidates(a)[0][0]=="11"


@pytest.mark.parametrize("mode",["ctc","segmented"])
def test_real_pytorch_training_and_inference(mode,tmp_path,monkeypatch):
    import cv2
    from taiwan_lottery.ml.layout.vision.training import fit,infer
    monkeypatch.setenv("LAYOUT_OCR_DATA_DIR",str(tmp_path))
    samples=[]
    for i,label in enumerate([1,2,3,4,5,6,7,8]):
        im=cell(str(label))
        cv2.imwrite(str(tmp_path/f"{i}.png"),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        samples.append({"label":label,"crop_path":f"{i}.png"})
    net,losses=fit(samples,mode,epochs=1)
    assert losses and np.isfinite(losses).all()
    assert len(infer(net,mode,samples))==len(samples)


def test_tesseract_real_inference_and_hidden_cell_preservation(monkeypatch):
    from taiwan_lottery.ml.layout.vision.engines import TesseractRecognizer
    recognizer=TesseractRecognizer()
    if recognizer.status!="ready":
        pytest.skip("system Tesseract unavailable")
    r=recognizer.recognize(variants(cell()),100)
    assert r.candidates[0]["value"]==73,r
    from taiwan_lottery.ml.layout.vision import service
    monkeypatch.setattr(service,"engines",lambda:[recognizer])
    # One OPENED paper cell + a solid dark unopened cell.
    board=np.full((80,160,3),45,np.uint8)
    import cv2
    board[6:74,6:74]=cv2.cvtColor(cv2.resize(cell("1"),(68,68)),cv2.COLOR_RGB2BGR)
    result=service.recognize_board(b"original",board,1,2,[],retain=False)
    assert result["proposals"][1]["state"]=="UNOPENED"
    assert result["cells"][1] is None
    assert result["capture_id"] is None


def test_lazy_engine_loading_and_isolated_failure(monkeypatch,tmp_path):
    from taiwan_lottery.ml.layout.vision import registry
    monkeypatch.setenv("LAYOUT_OCR_MODEL_DIR",str(tmp_path))
    registry.reset()
    first=registry.engines()
    assert registry.engines() is first
    assert len(first)==9
    assert any(x.name=="trocr_small" and x.status in ("dependency_missing","model_missing") for x in first)
    registry.reset()


def test_benchmark_freezes_selection_before_test_truth():
    from taiwan_lottery.ml.layout.vision.benchmark import benchmark
    data=[]
    for b in range(12):
        for i,label in enumerate((1,2)):
            raw=output("tesseract",[(label,.9),(3-label,.1)])
            raw["quality"]["engine_fingerprint"]="fixed-version"
            data.append({"id":b*2+i,"capture_id":str(b),"source_board":str(b),"image_hash":str(b),
                "state":"OPENED","label":label,"rows":1,"cols":2,"quality":{},
                "scenarios":["clean"],"raw_outputs":{"raw_engines":[raw]}})
    a=benchmark(data)
    assert a["status"]=="ready"
    assert a["best_method"]["constrained"]
    test_ids=set(a["split_ids"]["test"])
    import copy
    changed=copy.deepcopy(data)
    for sample in changed:
        if sample["id"] in test_ids:
            sample["label"]=3-sample["label"]
    b=benchmark(changed)
    assert a["best_method"]==b["best_method"]
    assert a["calibration"]==b["calibration"]
    selected=next(k for k in a["methods"] if k.endswith(":constraint"))
    assert a["methods"][selected]["test"]["exact_accuracy"]==1
    assert b["methods"][selected]["test"]["exact_accuracy"]==0


def test_calibration_rejects_stale_engine_version():
    r=output("a",[(73,.8)])
    r["quality"]["engine_fingerprint"]="new"
    calibration={"kind":"real_validation","engines":{"a":{
        "samples":100,"source_groups":4,"fingerprint":"old",
        "exact_accuracy":.99,"character_accuracy":.99,"brier":.01}}}
    fused=RecognitionConsensus(calibration).fuse([r],{"a":"a"},{})
    assert fused["weight_mode"]=="equal_family_prior_heuristic"
