"""Explicit model preparation, never imported/run on application startup."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request
from .registry import model_root


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and p.name not in ("artifact.json","LICENSE_VERIFIED")}


def prepare(name):
    root=model_root();root.mkdir(parents=True,exist_ok=True)
    if name=="tesseract":
        revision="87416418657359cb625c412a48b6e1d6d41c29bd"
        location=root/"tesseract";location.mkdir(parents=True,exist_ok=True)
        url="https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/"+revision+"/eng.traineddata"
        urllib.request.urlretrieve(url,location/"eng.traineddata")
        manifest={"source":url,"revision":revision,"model":"eng","license":"Apache-2.0",
                  "files":hashes(location)}
        (location/"artifact.json").write_text(json.dumps(manifest,indent=2))
        return manifest
    if name in ("parseq-tiny","crnn"):
        from .scene import prepare_scene
        return prepare_scene(root,name)
    if name=="easyocr":
        import easyocr
        easyocr.Reader(["en"],gpu=False,detector=False,recognizer=True,
            model_storage_directory=str(root/"easyocr"),download_enabled=True,verbose=False)
        location=root/"easyocr"
        manifest={"source":"https://github.com/JaidedAI/EasyOCR/releases/tag/v1.3",
                  "model":"english_g2","license":"Apache-2.0","files":hashes(location)}
    elif name.startswith("trocr"):
        from huggingface_hub import HfApi,snapshot_download
        repo="microsoft/"+name+"-printed"
        info=HfApi().model_info(repo)
        license_name=(info.card_data or {}).get("license")
        if license_name not in ("mit","apache-2.0","bsd-3-clause","bsd-2-clause"):
            raise ValueError("Model-card license is not verified permissive; model disabled")
        location=root/(name+"-printed")
        snapshot_download(repo,revision=info.sha,local_dir=str(location),
            ignore_patterns=["*.h5","*.msgpack","tf_model*","flax_model*"])
        manifest={"source":"https://huggingface.co/"+repo,"revision":info.sha,
                  "license":license_name,"files":hashes(location)}
        (location/"LICENSE_VERIFIED").write_text(license_name)
    else:
        if name not in ("PP-OCRv6_medium_rec","PP-OCRv6_small_rec","en_PP-OCRv5_mobile_rec"):
            raise ValueError("Unsupported official model")
        url="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/"+name+"_infer.tar"
        location=root/name
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/"model.tar"
            urllib.request.urlretrieve(url,archive)
            unpack=Path(tmp)/"unpacked";unpack.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(unpack,filter="data")
            candidates=list(unpack.rglob("inference.json"))
            if len(candidates)!=1:
                raise ValueError("Unexpected model archive layout")
            import shutil
            shutil.copytree(candidates[0].parent,location,dirs_exist_ok=True)
            manifest={"source":url,"model":name,"package_version":"paddleocr==3.7.0",
                      "license":"Apache-2.0","archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "files":hashes(location)}
    (location/"artifact.json").write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",choices=["tesseract","easyocr","trocr-small","trocr-base","PP-OCRv6_medium_rec",
                                       "PP-OCRv6_small_rec","en_PP-OCRv5_mobile_rec","parseq-tiny","crnn"])
    args=parser.parse_args()
    print(json.dumps(prepare(args.model),indent=2))
