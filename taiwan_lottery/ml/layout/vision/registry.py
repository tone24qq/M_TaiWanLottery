"""Lazy singleton backends. Failures isolated; no remote fetch from inference."""
from pathlib import Path
from threading import RLock
import os
import time
from .engines import TesseractRecognizer, EasyRecognizer, PaddleRecognizer, TrOCRRecognizer
from .base import BaseCellRecognizer, EngineResult

_lock=RLock()
_engines=None


class Unavailable(BaseCellRecognizer):
    def __init__(self,name,reason):
        self.name=name
        self.status="not_ready"
        self.reason=reason
    def recognize(self,*args,**kwargs):
        return EngineResult(self.name,status=self.status,reason=self.reason)


def model_root():
    return Path(os.getenv("LAYOUT_OCR_MODEL_DIR","model_artifacts/layout_ocr")).resolve()


def engines():
    global _engines
    with _lock:
        if _engines is not None:
            return _engines
        root=model_root()
        factories=[("tesseract",TesseractRecognizer),
                   ("easyocr",lambda:EasyRecognizer(root)),
                   *[("paddle_"+m.lower(),lambda m=m:PaddleRecognizer(root,m)) for m in (
                       "PP-OCRv6_medium_rec","PP-OCRv6_small_rec","en_PP-OCRv5_mobile_rec")],
                   ("trocr_small",lambda:TrOCRRecognizer(root,"small")),
                   ("trocr_base",lambda:TrOCRRecognizer(root,"base"))]
        from .digit import DigitRecognizer
        factories += [("digit_ctc",lambda:DigitRecognizer(root,"ctc")),
                      ("digit_segmented",lambda:DigitRecognizer(root,"segmented"))]
        _engines=[]
        for name,factory in factories:
            start=time.perf_counter()
            try:
                engine=factory()
            except Exception as exc:
                engine=Unavailable(name,type(exc).__name__+": "+str(exc)[:250])
            from importlib.metadata import version,PackageNotFoundError
            package={"easyocr":"easyocr","paddle":"paddleocr","trocr":"transformers",
                     "digit":"torch","tesseract":"pytesseract"}.get(engine.family)
            try:
                engine.version=version(package) if package else None
            except PackageNotFoundError:
                engine.version=None
            import hashlib
            artifact = root/("easyocr" if engine.family=="easyocr" else
                name.replace("trocr_","trocr-")+"-printed" if engine.family=="trocr" else
                engine.name)
            artifact=getattr(engine,"artifact_location",artifact)
            manifest=artifact/("manifest.json" if engine.family=="digit" else "artifact.json")
            engine.artifact_digest=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
            engine.load_seconds=time.perf_counter()-start
            _engines.append(engine)
        return _engines


def reset():
    global _engines
    with _lock:
        _engines=None
