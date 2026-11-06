"""Bounded CPU fine-tuning, board-disjoint comparisons and explicit promotion."""
import hashlib
import json
from pathlib import Path
import time
import uuid
import cv2
import numpy as np
from .dataset import grouped_split, stage
from .digit import network, tensor_image, ctc_candidates
from .preprocess import segment_digits
from .service import storage_root
from .registry import model_root


def augment(rgb,rng):
    """Training only: perspective, glare, scratches, motion/defocus, JPEG, dirt."""
    h,w=rgb.shape[:2]
    src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dst=src+rng.uniform(-3,3,(4,2)).astype("float32")
    image=cv2.warpPerspective(rgb,cv2.getPerspectiveTransform(src,dst),(w,h),borderMode=cv2.BORDER_REPLICATE)
    angle=float(rng.uniform(-8,8))
    image=cv2.warpAffine(image,cv2.getRotationMatrix2D((w/2,h/2),angle,1),(w,h),borderMode=cv2.BORDER_REPLICATE)
    image=np.clip(image.astype("float32")*rng.uniform(.65,1.3)+rng.normal(0,4,image.shape),0,255).astype("uint8")
    illumination=np.linspace(float(rng.uniform(.6,1.0)),1,w)[None,:,None]
    image=np.clip(image*illumination,0,255).astype("uint8")
    if rng.random()<.35:
        image=cv2.GaussianBlur(image,(3,3),float(rng.uniform(.3,1.2)))
    if rng.random()<.2:
        kernel=np.zeros((3,3),np.float32);kernel[1,:]=1/3
        image=cv2.filter2D(image,-1,kernel)
    if rng.random()<.25:
        overlay=image.copy()
        cv2.circle(overlay,(int(rng.integers(w)),int(rng.integers(h))),max(2,h//4),(255,255,255),-1)
        image=cv2.addWeighted(image,.7,overlay,.3,0)
    for _ in range(int(rng.integers(3))):
        x,y=int(rng.integers(w)),int(rng.integers(h))
        cv2.line(image,(x,y),(min(w-1,x+10),min(h-1,y+4)),(170,170,170),1)
    # Partial masking and dirty background are weak perturbations, never test truth.
    if rng.random()<.15:
        image[:int(rng.integers(1,max(2,h//8))),:]=rng.integers(80,180)
    small=cv2.resize(image,(max(12,w//2),max(12,h//2)))
    image=cv2.resize(small,(w,h))
    ok,encoded=cv2.imencode(".jpg",cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(45,95))])
    return cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB) if ok else image


def read_image(sample):
    path=(storage_root()/sample["crop_path"]).resolve()
    if not path.is_relative_to(storage_root()):
        raise ValueError("Crop outside dataset root")
    image=cv2.imread(str(path))
    if image is None:
        raise ValueError("Missing dataset crop")
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def fit(samples,mode,epochs=5):
    """Both architectures have real trainable weights; no external model code."""
    import torch
    from torch import nn
    rng=np.random.default_rng(42)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        net=network(mode)
        x=[];y=[]
        for sample in samples:
            im=augment(read_image(sample),rng)
            text=str(sample["label"])
            if mode=="ctc":
                x.append(tensor_image(im));y.append(text)
            else:
                parts=segment_digits(im)
                if len(parts)!=len(text):
                    continue
                for part,digit in zip(parts,text):
                    x.append(tensor_image(part,width=32));y.append(int(digit))
        if not x:
            raise ValueError("No valid segmented training examples")
        optimizer=torch.optim.Adam(net.parameters(),lr=.002)
        losses=[]
        for epoch in range(epochs):
            for start in range(0,len(x),32):
                batch=torch.stack(x[start:start+32])
                optimizer.zero_grad()
                logits=net(batch)
                if mode=="ctc":
                    labels=y[start:start+32]
                    target=torch.tensor([int(d) for label in labels for d in label],dtype=torch.long)
                    loss=nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0,1),target,
                        torch.full((len(labels),),logits.shape[1],dtype=torch.long),
                        torch.tensor([len(s) for s in labels]),blank=10,zero_infinity=True)
                else:
                    loss=nn.functional.cross_entropy(logits,torch.tensor(y[start:start+32]))
                loss.backward();optimizer.step()
                losses.append(float(loss.detach()))
        net.eval()
        return net,losses


def infer(net,mode,samples):
    import torch
    results=[]
    for sample in samples:
        im=read_image(sample)
        with torch.inference_mode():
            if mode=="ctc":
                probs=torch.softmax(net(tensor_image(im)[None]),-1)[0].numpy()
                candidates=ctc_candidates(probs,5)
                guess=next((s for s,p in candidates if s.isdigit()),"")
            else:
                parts=segment_digits(im)
                guess="".join(str(int(net(tensor_image(p,width=32)[None]).argmax(-1))) for p in parts)
        results.append(guess)
    return results


def train(samples,job_id,epochs=5):
    from .benchmark import score
    from .registry import reset
    import torch
    root=model_root();root.mkdir(parents=True,exist_ok=True)
    job=root/("training-"+job_id+".json")
    started=time.perf_counter()
    try:
        samples=[s for s in samples if s["label"] is not None and s["state"]=="OPENED"]
        readiness=stage(samples);split=grouped_split(samples)
        if readiness=="not_ready" or not split["validation"] or not split["test"]:
            report={"status":"not_ready","reason":"requires_500_cells_and_three_source_boards"}
        else:
            # No previous specialist outputs may act as the baseline.
            baseline_names=sorted({x["engine"] for s in split["test"] for x in s["raw_outputs"]["raw_engines"]
                                   if x["status"]=="ready" and not x["engine"].startswith("digit_")})
            baseline=score(split["test"],baseline_names,True)["exact_accuracy"]
            reports={}
            for mode in ("ctc","segmented"):
                net,losses=fit(split["train"],mode,epochs)
                validation=infer(net,mode,split["validation"])
                testing=infer(net,mode,split["test"])
                accuracy=lambda guesses,data:sum(g==str(s["label"]) for g,s in zip(guesses,data))/len(data)
                val,test=accuracy(validation,split["validation"]),accuracy(testing,split["test"])
                # Test is a release gate, not repeatedly searched for a configuration.
                promoted=readiness=="benchmark_eligible" and baseline is not None and test>baseline
                folder=root/("digit_"+mode);folder.mkdir(exist_ok=True)
                weights=folder/"weights.pt"
                temp=folder/"weights.tmp"
                torch.save(net.state_dict(),temp);temp.replace(weights)
                manifest={"architecture":mode,"stage":readiness,"promoted":promoted,
                    "sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "training_sources":sorted({s["source_board"] for s in split["train"]}),
                    "split_sample_ids":{k:[s["id"] for s in v] for k,v in split.items()},
                    "validation_accuracy":val,"test_accuracy":test,"ocr_test_accuracy":baseline,
                    "epochs":epochs,"seed":42,"final_loss":losses[-1],"job_id":job_id}
                (folder/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
                reports[mode]=manifest
            report={"status":"complete","models":reports,"elapsed_seconds":time.perf_counter()-started,
                    "selection":"validation_comparison; independent test release gate",
                    "best_sequence_strategy":max(reports,key=lambda m:reports[m]["validation_accuracy"])}
            reset()
        job.write_text(json.dumps(report),encoding="utf-8")
    except Exception as exc:
        report={"status":"failed","reason":type(exc).__name__+": "+str(exc)}
        job.write_text(json.dumps(report),encoding="utf-8")
    return report
