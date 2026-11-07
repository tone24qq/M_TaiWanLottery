"""Real cell-specific PyTorch CTC sequence and segmented digit models.

No MNIST weights. No fixed 1..100 classes. Untrained models cannot vote.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
from .base import BaseCellRecognizer
from .engines import fuse_views


def tensor_image(rgb, width=96, height=32):
    import cv2
    import torch
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) if rgb.ndim==3 else rgb
    gray=cv2.resize(gray,(width,height),interpolation=cv2.INTER_AREA)
    return torch.from_numpy(gray.astype("float32")/127.5-1).unsqueeze(0)


def network(mode):
    from torch import nn
    class DigitNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1),nn.ReLU())
            self.head=nn.Linear(64,11 if mode=="ctc" else 10)
        def forward(self,x):
            x=self.encoder(x)
            if mode=="ctc":
                return self.head(x.mean(dim=2).transpose(1,2))
            return self.head(x.mean(dim=(2,3)))
    return DigitNet()


def ctc_candidates(probabilities, beam_width=5):
    """CTC prefix beam search, including repeated digits separated by blank."""
    # prefix -> (blank mass, nonblank mass), computed in float64.
    prefixes={"":(1.0,0.0)}
    for row in probabilities:
        next_beams={}
        def add(prefix,blank=0.0,text=0.0):
            b,t=next_beams.get(prefix,(0.0,0.0))
            next_beams[prefix]=(b+blank,t+text)
        for prefix,(blank,text) in prefixes.items():
            add(prefix,blank=(blank+text)*float(row[10]))
            for digit in range(10):
                char=str(digit);p=float(row[digit])
                if prefix.endswith(char):
                    add(prefix,text=text*p)
                    if len(prefix)<3:
                        add(prefix+char,text=blank*p)
                elif len(prefix)<3:
                    add(prefix+char,text=(blank+text)*p)
        prefixes=dict(sorted(next_beams.items(),key=lambda item:-sum(item[1]))[:beam_width])
    return [(prefix,float(sum(mass))**(1/max(1,len(probabilities)))) for prefix,mass in sorted(prefixes.items(),key=lambda item:-sum(item[1]))]


class DigitRecognizer(BaseCellRecognizer):
    family="digit"

    def __init__(self,root,mode):
        self.mode=mode;self.name="digit_"+mode
        folder=root/self.name
        self.artifact_location=folder
        manifest=folder/"manifest.json"
        if not manifest.exists():
            self.status,self.reason="model_missing","requires_500_real_corrected_cells_and_independent_validation"
            return
        info=json.loads(manifest.read_text())
        self.training_sources=info.get("training_sources",[])
        self.experimental=not info.get("promoted",False)
        if self.experimental and not __import__("os").getenv("LAYOUT_OCR_EXPERIMENTAL_DIGIT"):
            self.status,self.reason="not_ready","experimental_not_better_than_ocr_on_independent_test"
            return
        weights=folder/"weights.pt"
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=info["sha256"]:
            raise ValueError("Digit artifact checksum mismatch")
        import torch
        self.net=network(mode)
        self.net.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
        self.net.eval()
        self.artifact_bytes=weights.stat().st_size
        self.status="ready"

    def recognize(self,variants,maximum,top_k=5):
        import torch
        from .preprocess import segment_digits
        readings=[]
        for view in variants.values():
            if self.mode=="ctc":
                with torch.inference_mode():
                    probs=torch.softmax(self.net(tensor_image(view)[None]),dim=-1)[0].numpy()
                readings.extend(ctc_candidates(probs,top_k))
            else:
                rgb=view if view.ndim==3 else np.repeat(view[:,:,None],3,axis=2)
                parts=segment_digits(rgb)
                if not 1<=len(parts)<=3:
                    readings.append(("",0));continue
                sequences=[("",1.0)]
                for part in parts:
                    with torch.inference_mode():
                        probabilities=torch.softmax(self.net(tensor_image(part,width=32)[None]),-1)[0]
                    candidates=torch.topk(probabilities,min(top_k,10))
                    sequences=sorted([(s+str(int(d)),score*float(p)) for s,score in sequences
                                      for p,d in zip(candidates.values,candidates.indices)],
                                     key=lambda x:-x[1])[:top_k]
                readings.extend(sequences)
        # Beam hypotheses are alternatives within each view, not independent votes.
        result=fuse_views(self.name,readings or [("",0)],maximum,top_k,view_count=len(variants))
        result.quality["training_sources"]=self.training_sources
        return result
