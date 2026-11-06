"""Source-board split: all crops of an original image must stay together."""
import hashlib


def grouped_split(samples):
    # Union by physical board ID OR exact original image hash (even if a user
    # accidentally assigns a second board ID). Never split cells independently.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:
            parent[x]=find(parent[x])
        return parent[x]
    for sample in samples:
        a,b=find("board:"+sample["source_board"]),find("image:"+sample["image_hash"])
        parent[b]=a
    groups={}
    for sample in samples:
        root=find("board:"+sample["source_board"])
        groups.setdefault(root,[]).append(sample)
    # Deterministic group-hash order; expose manifests to reproduce a training run.
    keys=sorted(groups,key=lambda k:hashlib.sha256(k.encode()).hexdigest())
    if len(keys)<3:
        return {"train":[],"validation":[],"test":[],"reason":"at_least_three_independent_boards_required"}
    train_end=max(1,int(len(keys)*.7))
    validation_end=max(train_end+1,int(len(keys)*.85))
    validation_end=min(validation_end,len(keys)-1)
    result={"train":[],"validation":[],"test":[]}
    for i,key in enumerate(keys):
        split="train" if i<train_end else "validation" if i<validation_end else "test"
        result[split].extend(groups[key])
    return result


def stage(samples):
    count=sum(s.get("label") is not None and s.get("state")=="OPENED" for s in samples)
    return "not_ready" if count<500 else "experimental" if count<2000 else "benchmark_eligible"
