"""Official-package adapters. No implicit downloads in request paths."""
from collections import defaultdict
from pathlib import Path
import os
import shutil
import time
import importlib.util
from .base import BaseCellRecognizer, EngineResult, numeric


def fuse_views(name, readings, maximum, top_k=5, view_count=None, view_names=None, glyph_count=None):
    scores=defaultdict(list)
    lengths=defaultdict(set)
    for text,score in readings:
        number=numeric(text,maximum)
        if number is not None:
            scores[number].append(max(0.0,min(float(score),1.0)))
            lengths[number].add(len(str(text).strip()))
    # A preprocessing family contributes one engine result.
    # Repeated views cannot multiply an engine's external voting weight.
    # ROI proposals are alternatives, not independent votes. Keep the strongest
    # two native observations of each value, rather than letting nine correlated
    # bad thresholdings suppress one clear, high-confidence text crop.
    candidates=[{"value":number,"score":float(sum(sorted(values,reverse=True)[:2])/min(2,len(values))),
                 "view_support":len(values)}
                for number,values in scores.items()]
    if candidates and glyph_count:
        highest=max(c["score"] for c in candidates)
        compatible=any(glyph_count in lengths[c["value"]] and c["score"]>=highest*.95 for c in candidates)
        if compatible:
            for candidate in candidates:
                if glyph_count not in lengths[candidate["value"]]:
                    candidate["score"]*=.94
    candidates.sort(key=lambda x:(-x["score"],-x["view_support"],x["value"]))
    quality={"views":len(readings),"score_kind":"native_proposal_evidence_not_probability",
             "aggregation":"mean_of_best_two_views_per_value","visible_digit_count":glyph_count}
    if view_names:
        quality["view_results"]=[{"view":label,"raw_text":str(t),"score":float(score)}
                                 for label,(t,score) in zip(view_names,readings)]
    return EngineResult(name,candidates[:top_k],[str(t) for t,_ in readings],quality)


def verify_cache(location):
    import hashlib,json
    manifest=location/"artifact.json"
    if not manifest.exists():
        raise ValueError("Prepared artifact manifest required; run explicit model preparation")
    info=json.loads(manifest.read_text())
    for name,expected in info["files"].items():
        path=(location/name).resolve()
        if not path.is_relative_to(location.resolve()) or not path.is_file():
            raise ValueError("Missing or unsafe cached artifact")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError("Cached model checksum mismatch")


class TesseractRecognizer(BaseCellRecognizer):
    name=family="tesseract"

    def __init__(self):
        self.tessdata=None
        root=Path(os.getenv("LAYOUT_OCR_MODEL_DIR","model_artifacts/layout_ocr"))
        location=root/"tesseract"
        if (location/"artifact.json").exists():
            verify_cache(location)
            self.tessdata=location.resolve()
            self.artifact_location=location
            self.artifact_bytes=(location/"eng.traineddata").stat().st_size
        self.status="ready" if shutil.which("tesseract") else "dependency_missing"
        self.reason=None if self.status=="ready" else "tesseract_binary_missing"

    def recognize(self, variants, maximum, top_k=5):
        import pytesseract
        from PIL import Image
        from tempfile import TemporaryDirectory
        # One multipage TIFF / process, instead of nine process launches per cell.
        images=[Image.fromarray(v).convert("RGB") for v in variants.values()]
        with TemporaryDirectory(prefix="cell-ocr-") as directory:
            path=Path(directory)/"views.tiff"
            images[0].save(path,save_all=True,append_images=images[1:],dpi=(300,300))
            config="--psm 7 -c tessedit_char_whitelist=0123456789"
            if self.tessdata:
                config+=' --tessdata-dir "'+str(self.tessdata)+'" -l eng'
            data=pytesseract.image_to_data(str(path),config=config,
                output_type=pytesseract.Output.DICT,timeout=8)
        pages=[[] for _ in images]
        for text,confidence,page in zip(data["text"],data["conf"],data["page_num"]):
            if text.strip() and float(confidence)>=0 and 1<=int(page)<=len(pages):
                pages[int(page)-1].append((text.strip(),float(confidence)/100))
        readings=[("".join(text for text,_ in row),sum(c for _,c in row)/len(row)) if row else ("",0)
                  for row in pages]
        return fuse_views(self.name,readings,maximum,top_k,view_names=list(variants),glyph_count=getattr(variants,"glyph_count",None))


class EasyRecognizer(BaseCellRecognizer):
    name=family="easyocr"

    def __init__(self, root):
        if not importlib.util.find_spec("easyocr"):
            self.status,self.reason="dependency_missing","install_requirements_ocr"
            return
        if not (root/"easyocr"/"english_g2.pth").exists():
            self.status,self.reason="model_missing","run_explicit_model_preparation"
            return
        self.artifact_location=root/"easyocr"
        verify_cache(self.artifact_location)
        import easyocr
        self.reader=easyocr.Reader(["en"],gpu=False,detector=False,recognizer=True,
            model_storage_directory=str(root/"easyocr"),download_enabled=False,verbose=False)
        self.status="ready"
        self.artifact_bytes=(root/"easyocr"/"english_g2.pth").stat().st_size

    def recognize(self, variants, maximum, top_k=5):
        import cv2
        readings=[];names=[]
        for name,view in variants.items():
            gray=cv2.cvtColor(view,cv2.COLOR_RGB2GRAY) if view.ndim==3 else view
            rows=self.reader.recognize(gray,allowlist="0123456789",decoder="beamsearch",
                beamWidth=5,detail=1,paragraph=False)
            found=[(row[1],row[2]) for row in rows] or [("",0)]
            readings.extend(found);names.extend([name]*len(found))
        return fuse_views(self.name,readings,maximum,top_k,view_names=names,glyph_count=getattr(variants,"glyph_count",None))


class PaddleRecognizer(BaseCellRecognizer):
    family="paddle"

    def __init__(self, root, model):
        self.name="paddle_"+model.lower()
        if not importlib.util.find_spec("paddleocr") or not importlib.util.find_spec("paddle"):
            self.status,self.reason="dependency_missing","install_paddle_requirements"
            return
        location=root/model
        if not location.is_dir() or not (location/"inference.json").exists():
            self.status,self.reason="model_missing","run_explicit_model_preparation"
            return
        self.artifact_location=location
        verify_cache(location)
        from paddleocr import TextRecognition
        self.reader=TextRecognition(model_name=model,model_dir=str(location),device="cpu",enable_mkldnn=False)
        self.status="ready"
        self.artifact_bytes=sum(p.stat().st_size for p in location.rglob("*") if p.is_file())

    def recognize(self, variants, maximum, top_k=5):
        import cv2
        readings=[]
        for view in variants.values():
            bgr=cv2.cvtColor(view,cv2.COLOR_RGB2BGR) if view.ndim==3 else cv2.cvtColor(view,cv2.COLOR_GRAY2BGR)
            for row in self.reader.predict(input=bgr,batch_size=1):
                readings.append((row.get("rec_text",""),row.get("rec_score",0)))
        return fuse_views(self.name,readings or [("",0)],maximum,top_k)


class TrOCRRecognizer(BaseCellRecognizer):
    family="trocr"

    def __init__(self, root, size):
        self.name="trocr_"+size
        if not importlib.util.find_spec("transformers"):
            self.status,self.reason="dependency_missing","install_requirements_ocr"
            return
        location=root/("trocr-"+size+"-printed")
        # This manifest must be written by explicit preparation after license verification.
        if not (location/"LICENSE_VERIFIED").exists():
            self.status,self.reason="model_missing","verified_local_snapshot_required"
            return
        self.artifact_location=location
        verify_cache(location)
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.processor=TrOCRProcessor.from_pretrained(str(location),local_files_only=True)
        self.reader=VisionEncoderDecoderModel.from_pretrained(str(location),local_files_only=True).eval()
        self.status="ready"
        self.artifact_bytes=sum(p.stat().st_size for p in location.rglob("*") if p.is_file())

    def recognize(self, variants, maximum, top_k=5):
        import torch
        import numpy as np
        from PIL import Image
        readings=[]
        for view in variants.values():
            im=Image.fromarray(view).convert("RGB")
            pixels=self.processor(images=im,return_tensors="pt").pixel_values
            with torch.inference_mode():
                output=self.reader.generate(pixels,num_beams=top_k,num_return_sequences=top_k,
                    max_new_tokens=5,return_dict_in_generate=True,output_scores=True)
            texts=self.processor.batch_decode(output.sequences,skip_special_tokens=True)
            scores=torch.softmax(output.sequences_scores,dim=0).tolist()
            readings.extend(zip(texts,scores))
        return fuse_views(self.name,readings,maximum,top_k,view_count=len(variants))
