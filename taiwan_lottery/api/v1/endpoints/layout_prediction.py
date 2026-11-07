"""Local layout workflow using the existing async SQLAlchemy session."""
import hashlib
import json
from threading import Lock
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from taiwan_lottery.api.deps import get_db
from taiwan_lottery.db.models.layout import LayoutHistory, LayoutRun
from taiwan_lottery.ml.layout.schema import PredictionRequest, HistoryRequest, EvaluationRequest, Resolution
from taiwan_lottery.ml.layout.registry import CLASSES
from taiwan_lottery.ml.layout.predictor import predict
from taiwan_lottery.ml.layout.backtesting import evaluate, metrics
from taiwan_lottery.ml.layout.ocr import recognize

router = APIRouter()
# Serialize CPU training/RNG scopes inside each application process.
_compute_lock = Lock()


def compute(fn, *args):
    with _compute_lock:
        return fn(*args)


async def corpus(db, rows, cols):
    found = (await db.scalars(select(LayoutHistory).where(
        LayoutHistory.rows==rows, LayoutHistory.cols==cols).order_by(LayoutHistory.id))).all()
    if len(found)>200:
        raise HTTPException(422,"Interactive budget is 200 same-size layouts; use offline training for larger corpora")
    return found


async def prior_evaluation(db, rows, cols, history_ids):
    records = (await db.scalars(select(LayoutRun).where(
        LayoutRun.kind=="evaluation",LayoutRun.rows==rows,LayoutRun.cols==cols
    ).order_by(LayoutRun.id.desc()))).all()
    # Reuse only an evaluation based on exactly the current corpus, never future/different data.
    for record in records:
        if record.request.get("history_ids")==history_ids:
            return record.result.get("votes",[])
    return []


@router.get("/models")
async def models(rows: int=10, cols: int=10, db: AsyncSession=Depends(get_db)):
    if not 1<=rows*cols<=400 or not 1<=rows<=100 or not 1<=cols<=100:
        raise HTTPException(422,"Invalid dimensions")
    history = await corpus(db,rows,cols)
    result = []
    for cls in CLASSES:
        import importlib.util
        available = not cls.package or importlib.util.find_spec(cls.package) is not None
        ready = available and len(history)>=cls.minimum
        result.append({"model":cls.name,"status":"ready" if ready else "not_ready",
                       "reason":None if ready else ("insufficient_complete_layouts" if available else "dependency_unavailable"),
                       "minimum_complete_layouts":cls.minimum,"available_complete_layouts":len(history)})
    return result


@router.post("/history")
async def add_history(body: HistoryRequest, db: AsyncSession=Depends(get_db)):
    fingerprint = hashlib.sha256(json.dumps([body.rows,body.cols,body.cells]).encode()).hexdigest()
    record = LayoutHistory(rows=body.rows,cols=body.cols,cells=body.cells,
                           source=body.source,fingerprint=fingerprint)
    db.add(record)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409,"Duplicate source or complete layout")
    return {"id":record.id,"status":"confirmed"}


@router.post("/predict")
@router.post("/discuss")
async def discussion(body: PredictionRequest, db: AsyncSession=Depends(get_db)):
    history = await corpus(db,body.rows,body.cols)
    ids = [h.id for h in history]
    votes = await prior_evaluation(db,body.rows,body.cols,ids)
    result = await run_in_threadpool(compute,predict,body,[h.cells for h in history],votes,votes)
    record = LayoutRun(kind="prediction",rows=body.rows,cols=body.cols,
        request={**body.model_dump(),"history_ids":ids},result=result)
    db.add(record)
    await db.flush()
    return {"run_id":record.id,**result}


@router.post("/evaluation")
async def run_evaluation(body: EvaluationRequest, db: AsyncSession=Depends(get_db)):
    history = await corpus(db,body.rows,body.cols)
    if body.rows*body.cols>400:
        raise HTTPException(422,"At most 400 cells")
    result = await run_in_threadpool(compute,evaluate,[h.cells for h in history],
        body.rows,body.cols,body.seed,body.revealed_fraction,body.max_folds)
    record = LayoutRun(kind="evaluation",rows=body.rows,cols=body.cols,
        request={**body.model_dump(),"history_ids":[h.id for h in history]},result=result)
    db.add(record)
    await db.flush()
    return {"run_id":record.id,**result}


@router.get("/evaluation")
async def latest_evaluation(rows: int=10, cols: int=10, db: AsyncSession=Depends(get_db)):
    record = await db.scalar(select(LayoutRun).where(LayoutRun.kind=="evaluation",
        LayoutRun.rows==rows,LayoutRun.cols==cols).order_by(LayoutRun.id.desc()).limit(1))
    return {"status":"not_ready","reason":"no_walk_forward_evaluation"} if record is None else {
        "run_id":record.id, "request":record.request, **record.result}


@router.get("/runs/{run_id}")
async def get_run(run_id: int, db: AsyncSession=Depends(get_db)):
    record = await db.get(LayoutRun,run_id)
    if record is None:
        raise HTTPException(404,"Run not found")
    return {"run_id":record.id,"request":record.request,"result":record.result,"outcome":record.outcome}


@router.post("/runs/{run_id}/resolve")
async def resolve(run_id: int, body: Resolution, db: AsyncSession=Depends(get_db)):
    record = await db.scalar(select(LayoutRun).where(LayoutRun.id==run_id).with_for_update())
    if record is None or record.kind!="prediction" or record.result.get("status")!="ready":
        raise HTTPException(404,"Unresolved prediction not found")
    if record.outcome is not None:
        raise HTTPException(409,"Outcome already recorded; original prediction cannot be rewritten")
    b = PredictionRequest(**{k:v for k,v in record.request.items() if k!="history_ids"})
    position = (body.row-1)*b.cols + body.col-1
    if body.row>b.rows or body.col>b.cols or position not in b.unknown:
        raise HTTPException(422,"Actual target must be in a previously hidden cell")
    actual = (body.row,body.col)
    outcomes = []
    for review in record.result["round2"]:
        before = metrics(review["before_ranking"],actual)
        after = metrics(review["after_ranking"],actual)
        outcomes.append({"model":review["model"],"before":before,"after":after,
                         "revision_result":after["mrr"]-before["mrr"],
                         "hit":bool(after["top1"])})
    record.outcome = {"actual":body.model_dump(),"votes":outcomes,
        "consensus":metrics(record.result["consensus"]["ranking"],actual)}
    return record.outcome


@router.post("/recognize")
async def image_recognition(file: UploadFile=File(...), rows: int=Form(...),
                            cols: int=Form(...), corners: str | None=Form(None), retain: bool=Form(False)):
    if rows<1 or cols<1 or rows>100 or cols>100 or rows*cols>400:
        raise HTTPException(422,"Invalid dimensions")
    data = await file.read(8*1024*1024+1)
    if len(data)>8*1024*1024:
        raise HTTPException(413,"Image exceeds 8 MiB")
    try:
        points = json.loads(corners) if corners else None
        return await run_in_threadpool(compute,recognize,data,rows,cols,points,retain)
    except (ValueError,TypeError,RuntimeError,OSError) as exc:
        raise HTTPException(422,str(exc))
