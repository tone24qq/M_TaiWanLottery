"""Pinned official PARSeq/CRNN adapters, benchmark-only without validation."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
from .base import BaseCellRecognizer
from .engines import fuse_views,verify_cache

REVISION="1902db043c029a7e03a3818c616c06600af574be"
REPOSITORY="https://github.com/baudm/parseq"
WEIGHTS={
    "parseq-tiny":("parseq_tiny-e7a21b54.pt","e7a21b543c98e67414a584c93b1dbb71c26e9463ae4aaa391d54e833660a4711"),
    "crnn":("crnn-679d0e31.pt","679d0e315af22d813c6d5a86287ee5705a9a8dfe37378349359442b8b8953264")}


class SceneRecognizer(BaseCellRecognizer):
    family="scene_text"

    def __init__(self,root,name):
        self.name=name.replace("-","_")
        self.artifact_location=root/name
        location=self.artifact_location
        if not (location/"artifact.json").exists():
            self.status,self.reason="model_missing","run_explicit_scene_model_preparation"
            return
        dependencies=os.getenv("LAYOUT_OCR_SCENE_DEPENDENCIES")
        if dependencies and dependencies not in sys.path:
            sys.path.insert(0,dependencies)
        try:
            import timm,pytorch_lightning,nltk
        except ImportError:
            self.status,self.reason="dependency_missing","install_requirements_scene_ocr"
            return
        verify_cache(location)
        info=json.loads((location/"artifact.json").read_text())
        if info.get("license")!="Apache-2.0" or info.get("revision")!=REVISION:
            raise ValueError("Unverified scene-text provenance")
        source=root/"scene_text"/("parseq-"+REVISION)
        if not source.is_dir():
            self.status,self.reason="model_missing","pinned_official_source_missing"
            return
        source_manifest=json.loads((root/"scene_text"/"source.json").read_text())
        for relative,digest in source_manifest["files"].items():
            path=(source/relative).resolve()
            if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError("Scene-text source checksum mismatch")
        if str(source) not in sys.path:sys.path.insert(0,str(source))
        import torch
        from strhub.models.utils import create_model
        from torchvision import transforms
        # No download: create architecture locally and load verified state dictionary.
        self.reader=create_model(name,pretrained=False).eval()
        state=torch.load(location/"weights.pt",map_location="cpu",weights_only=True)
        target=self.reader.model if name.startswith("parseq") else self.reader
        target.load_state_dict(state)
        self.transform=transforms.Compose([
            transforms.Resize(self.reader.hparams.img_size,transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),transforms.Normalize(.5,.5)])
        self.version=REVISION
        self.artifact_bytes=(location/"weights.pt").stat().st_size
        self.status="benchmark_only"
        self.reason="requires_independent_real_validation_before_runtime_promotion"

    def recognize(self,views,maximum,top_k=5):
        import torch
        from PIL import Image
        readings=[];names=[]
        started=time.perf_counter()
        for name,view in views.items():
            x=self.transform(Image.fromarray(view).convert("RGB")).unsqueeze(0)
            with torch.inference_mode():
                probabilities=self.reader(x,max_length=3).softmax(-1)
            texts,scores=self.reader.tokenizer.decode(probabilities)
            candidates=digit_lattice(probabilities[0],self.reader.tokenizer,self.name=="crnn",top_k)
            candidates.insert(0,(texts[0],float(scores[0].mean())))
            readings.extend(candidates);names.extend([name]*len(candidates))
        result=fuse_views(self.name,readings,maximum,top_k,view_names=names,glyph_count=getattr(views,"glyph_count",None))
        result.quality.update(runtime_policy="benchmark_only",model_version=REVISION,
                              artifact_sha256=WEIGHTS["parseq-tiny" if self.name=="parseq_tiny" else "crnn"][1],
                              inference_ms=(time.perf_counter()-started)*1000)
        return result


def prepare_scene(root,name):
    """Author release + pinned Apache source; no private mirrors or source vendoring."""
    import urllib.request,tarfile,tempfile,shutil
    root=Path(root);shared=root/"scene_text";shared.mkdir(parents=True,exist_ok=True)
    source=shared/("parseq-"+REVISION)
    if not source.exists():
        url="https://codeload.github.com/baudm/parseq/tar.gz/"+REVISION
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/"source.tar.gz"
            urllib.request.urlretrieve(url,archive)
            with tarfile.open(archive) as tar:tar.extractall(shared,filter="data")
    license_text=(source/"LICENSE").read_text()
    if "Apache License" not in license_text:
        raise ValueError("Missing permissive upstream license")
    files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in source.rglob("*") if p.is_file() and p.suffix in (".py",".yaml")}
    (shared/"source.json").write_text(json.dumps({"repository":REPOSITORY,"revision":REVISION,"license":"Apache-2.0","files":files}))
    location=root/name;location.mkdir(exist_ok=True)
    filename,expected=WEIGHTS[name]
    url=REPOSITORY+"/releases/download/v1.0.0/"+filename
    target=location/"weights.pt"
    if not target.exists():urllib.request.urlretrieve(url,target)
    actual=hashlib.sha256(target.read_bytes()).hexdigest()
    if actual!=expected:
        raise ValueError("Official scene-text weight checksum mismatch")
    from datetime import datetime,timezone
    manifest={"source":url,"model":name,"revision":REVISION,"license":"Apache-2.0",
              "prepared_at":datetime.now(timezone.utc).isoformat(),"sha256":actual,
              "model_size":target.stat().st_size,"runtime_policy":"benchmark_only",
              "files":{"weights.pt":actual}}
    (location/"artifact.json").write_text(json.dumps(manifest,indent=2))
    return manifest


def digit_lattice(probabilities,tokenizer,ctc,top_k):
    """Top-K from actual digit/EOS distributions; no invented numeric alternatives.

    PARSeq alternatives use the frozen decoder token lattice, not a claim of
    autoregressive beam-search probabilities. Scores remain uncalibrated.
    """
    import numpy as np
    values=probabilities.detach().cpu().numpy()
    mapping=tokenizer._stoi
    if ctc:
        from .digit import ctc_candidates
        digit=values[:,[mapping[str(i)] for i in range(10)]]
        blank=values[:,mapping[getattr(tokenizer,"BLANK","[B]")]][:,None]
        return ctc_candidates(np.concatenate([digit,blank],axis=1),top_k)
    beams=[("",1.0)];finished=[]
    eos=tokenizer.eos_id
    for row in values:
        expanded=[]
        for text,score in beams:
            if text:
                finished.append((text,(score*float(row[eos]))**(1/(len(text)+1))))
            if len(text)<3:
                expanded.extend((text+str(i),score*float(row[mapping[str(i)]])) for i in range(10))
        beams=sorted(expanded,key=lambda item:-item[1])[:top_k]
        if not beams:break
    return sorted(finished,key=lambda item:-item[1])[:top_k]
