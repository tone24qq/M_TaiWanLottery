"""OCR backend status, human dataset, validation benchmark and digit fine-tune."""
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4
import json
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from taiwan_lottery.api.deps import get_db
from taiwan_lottery.db.models.layout_digit import LayoutDigitSample
from taiwan_lottery.ml.layout.vision.registry import engines, model_root
from taiwan_lottery.ml.layout.vision.service import storage_root
from taiwan_lottery.ml.layout.vision.dataset import stage
from taiwan_lottery.ml.layout.vision.training import train
from taiwan_lottery.ml.layout.vision.benchmark import benchmark
from .layout_prediction import compute

router=APIRouter()


class CellCorrection(BaseModel):
    index: int=Field(ge=0,le=399)
    state: Literal["OPENED","UNOPENED","UNCERTAIN"]
    label: int | None=Field(default=None,ge=1,le=400)
    scenarios: list[Literal["clean","glare","scratch","blur","perspective","low-light"]]=Field(default_factory=list,max_length=6)


class Corrections(BaseModel):
    capture_id: UUID
    source_board: str=Field(min_length=1,max_length=200)
    cells: list[CellCorrection]=Field(min_length=1,max_length=400)
    auto_train: bool=True


def serialize(row):
    return {"id":row.id,"capture_id":row.capture_id,"source_board":row.source_board,
            "image_hash":row.image_hash,"crop_path":row.crop_path,"rows":row.rows,"cols":row.cols,
            "state":row.state,"label":row.label,"raw_outputs":row.raw_outputs,
            "quality":row.quality,"scenarios":row.quality.get("scenarios",[])}


async def samples(db):
    count=await db.scalar(select(func.count()).select_from(LayoutDigitSample))
    if count>20_000:
        raise HTTPException(422,"Interactive dataset budget exceeded; run offline training")
    return [serialize(r) for r in (await db.scalars(select(LayoutDigitSample).order_by(LayoutDigitSample.id))).all()]


@router.get("/ocr/models")
async def model_status():
    return await run_in_threadpool(compute,lambda:[e.metadata() for e in engines()])


@router.post("/ocr/corrections")
async def corrections(body:Corrections,tasks:BackgroundTasks,db:AsyncSession=Depends(get_db)):
    capture=str(body.capture_id)
    folder=storage_root()/"captures"/capture
    manifest=folder/"recognition.json"
    if not manifest.exists():
        raise HTTPException(404,"Capture not retained; enable OCR training capture before upload")
    original=json.loads(manifest.read_text(encoding="utf-8"))
    existing=await db.scalar(select(LayoutDigitSample.id).where(LayoutDigitSample.capture_id==capture).limit(1))
    if existing is not None:
        raise HTTPException(409,"Capture labels already confirmed; do not duplicate the same board")
    labels=[c.label for c in body.cells if c.label is not None]
    if len(set(c.index for c in body.cells))!=len(body.cells) or len(labels)!=len(set(labels)):
        raise HTTPException(422,"Duplicate cell indices or labels")
    for cell in body.cells:
        if cell.index>=len(original["cells"]) or (cell.label is not None and
                (cell.state!="OPENED" or cell.label>original["rows"]*original["cols"])):
            raise HTTPException(422,"Only confirmed OPENED cells may have a legal numeric label")
    for cell in body.cells:
        raw=original["proposals"][cell.index]
        db.add(LayoutDigitSample(capture_id=capture,cell_index=cell.index,source_board=body.source_board,
            image_hash=original["image_hash"],crop_path=f"captures/{capture}/{cell.index}.png",
            rows=original["rows"],cols=original["cols"],state=cell.state,label=cell.label,
            raw_outputs=raw,quality={**raw["quality"],"scenarios":cell.scenarios}))
    await db.flush()
    data=await samples(db)
    count=sum(s["label"] is not None for s in data)
    job_id=None
    if body.auto_train and count>=500 and count//500>(count-len(labels))//500:
        job_id=str(uuid4())
        tasks.add_task(run_in_threadpool,compute,train,data,job_id)
    return {"saved":len(body.cells),"labelled_cells":count,"stage":stage(data),"training_job":job_id}


@router.get("/ocr/dataset")
async def dataset_status(db:AsyncSession=Depends(get_db)):
    data=await samples(db)
    return {"samples":len(data),"labelled_cells":sum(s["label"] is not None for s in data),
            "source_boards":len({s["source_board"] for s in data}),"stage":stage(data)}


@router.post("/ocr/benchmark")
async def run_benchmark(db:AsyncSession=Depends(get_db)):
    data=await samples(db)
    report=await run_in_threadpool(compute,benchmark,data)
    root=model_root();root.mkdir(parents=True,exist_ok=True)
    (root/"benchmark.json").write_text(json.dumps(report),encoding="utf-8")
    # Apply validated weights/profile only with sufficient real validation coverage.
    if report["status"]=="ready" and all(r.get("samples",0)>=50 and r.get("source_groups",0)>=3
                                        for r in report["calibration"]["engines"].values()):
        (root/"calibration.json").write_text(json.dumps(report["calibration"]),encoding="utf-8")
        (root/"profile.json").write_text(json.dumps(report["best_method"]),encoding="utf-8")
    return report


@router.post("/ocr/digit/train")
async def train_digit(tasks:BackgroundTasks,db:AsyncSession=Depends(get_db)):
    data=await samples(db)
    if stage(data)=="not_ready":
        return {"status":"not_ready","reason":"requires_500_real_corrected_cells"}
    job_id=str(uuid4())
    tasks.add_task(run_in_threadpool,compute,train,data,job_id)
    return {"status":"scheduled","job_id":job_id}


@router.get("/ocr/digit/jobs/{job_id}")
async def training_status(job_id:UUID):
    path=model_root()/("training-"+str(job_id)+".json")
    return json.loads(path.read_text()) if path.exists() else {"status":"pending_or_unknown"}


@router.get("/ocr/benchmark")
async def last_benchmark():
    path=model_root()/"benchmark.json"
    return json.loads(path.read_text()) if path.exists() else {
        "status":"not_ready","accuracy_conclusion":"尚無真實照片 accuracy 結論"}
