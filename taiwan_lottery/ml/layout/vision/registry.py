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
        if os.getenv("LAYOUT_OCR_BENCHMARK")=="1":
            from .scene import SceneRecognizer
            factories += [(name.replace("-","_"),lambda name=name:SceneRecognizer(root,name))
                          for name in ("parseq-tiny","crnn")]
        _engines=[]
        for name,factory in factories:
            start=time.perf_counter()
            try:
                engine=factory()
            except Exception as exc:
                engine=Unavailable(name,type(exc).__name__+": "+str(exc)[:250])
            from importlib.metadata import version,PackageNotFoundError
            package={"easyocr":"easyocr","paddle":"paddleocr","trocr":"transformers",
                     "digit":"torch","tesseract":"pytesseract","scene_text":"torch"}.get(engine.family)
            try:
                engine.version=version(package) if package else None
            except PackageNotFoundError:
                engine.version=None
            import hashlib,json,importlib.util
            location_names={"paddle_pp-ocrv6_medium_rec":"PP-OCRv6_medium_rec",
                "paddle_pp-ocrv6_small_rec":"PP-OCRv6_small_rec",
                "paddle_en_pp-ocrv5_mobile_rec":"en_PP-OCRv5_mobile_rec",
                "trocr_small":"trocr-small-printed","trocr_base":"trocr-base-printed",
                "parseq_tiny":"parseq-tiny"}
            artifact=getattr(engine,"artifact_location",root/location_names.get(name,name))
            manifest=artifact/("manifest.json" if name.startswith("digit_") else "artifact.json")
            engine.artifact_digest=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
            try:
                info=json.loads(manifest.read_text()) if manifest.exists() else {}
                if not isinstance(info,dict):raise ValueError("Artifact manifest must be an object")
            except (OSError,ValueError):
                info={}
                engine.reason="artifact_manifest_invalid"

            expected=info.get("files",{})
            if not isinstance(expected,dict):expected={}
            weight_files=[p for p in artifact.glob("*") if p.is_file() and p.suffix in
                          (".pt",".pth",".bin",".safetensors",".traineddata",".pdiparams")]
            if not engine.artifact_bytes:
                engine.artifact_bytes=sum(p.stat().st_size for p in weight_files)
            downloaded=bool(weight_files) or bool(expected and all((artifact/f).is_file() for f in expected))
            checked=False
            if expected:
                checked=all((artifact/f).is_file() and (artifact/f).resolve().is_relative_to(artifact.resolve()) and
                            hashlib.sha256((artifact/f).read_bytes()).hexdigest()==digest for f,digest in expected.items())
            elif info.get("sha256") and (artifact/"weights.pt").exists():
                checked=hashlib.sha256((artifact/"weights.pt").read_bytes()).hexdigest()==info["sha256"]
            native=engine.name=="tesseract"
            package_name=("paddleocr" if name.startswith("paddle_") else
                          "transformers" if name.startswith("trocr_") else
                          "torch" if name.startswith("digit_") or name in ("parseq_tiny","crnn") else
                          "easyocr" if name=="easyocr" else "pytesseract")
            engine.readiness={
                "package_installed":importlib.util.find_spec(package_name) is not None,
                "model_downloaded":downloaded or (native and engine.status=="ready"),
                "checksum_verified":checked,
                "license_verified":info.get("license") in ("Apache-2.0","apache-2.0","MIT","mit","BSD-3-Clause","bsd-3-clause"),
                "inference_tested":False}
            engine.model_version=info.get("revision") or info.get("package_version") or engine.version
            engine.artifact_sha256=info.get("sha256") or info.get("archive_sha256") or (
                next(iter(expected.values())) if len(expected)==1 else engine.artifact_digest)
            engine.load_seconds=time.perf_counter()-start
            _engines.append(engine)
        return _engines


def reset():
    global _engines
    with _lock:
        _engines=None
