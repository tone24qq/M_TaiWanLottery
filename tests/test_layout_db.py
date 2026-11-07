"""Opt-in PostgreSQL integration, wrapped in a rollback-only transaction."""
import asyncio
import os
from io import BytesIO
import pytest

pytestmark = pytest.mark.skipif(os.getenv("LAYOUT_TEST_DATABASE")!="1",reason="requires migrated local PostgreSQL")


def test_layout_api_storage_evaluation_resolution_and_ocr(tmp_path,monkeypatch):
    monkeypatch.setenv("LAYOUT_OCR_DATA_DIR",str(tmp_path))
    asyncio.run(_scenario())


async def _scenario():
    import httpx
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from PIL import Image, ImageDraw, ImageFont
    from taiwan_lottery.main import app
    from taiwan_lottery.api.deps import get_db
    from taiwan_lottery.config import settings
    engine=create_async_engine(settings.DATABASE_URL)
    async with engine.connect() as connection:
        transaction=await connection.begin()
        session=async_sessionmaker(bind=connection,expire_on_commit=False)()
        async def test_db():
            yield session
        app.dependency_overrides[get_db]=test_db
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as c:
                    # Existing application routes and database-backed lottery/bingo
                    # reads must survive the OCR changes.
                    for page in ("/","/bingo","/predictions","/history/bingo","/analysis/lotto649"):
                        response=await c.get(page)
                        assert response.status_code==200,(page,response.text)
                    for game in ("bingo","lotto649","daily_cash","super_lotto"):
                        response=await c.get(f"/api/v1/{game}/draws")
                        assert response.status_code==200,(game,response.text)
                    r=await c.post("/api/v1/layout/predict",json={"rows":1,"cols":4,"cells":[1,None,None,None],"target":3})
                    assert r.status_code==200, r.text
                    prediction=r.json()
                    run_id=prediction["run_id"]
                    assert len(prediction["consensus"]["ranking"])==3
                    stored=(await c.get(f"/api/v1/layout/runs/{run_id}")).json()
                    assert stored["result"]["round_count"]==3
                    r=await c.post(f"/api/v1/layout/runs/{run_id}/resolve",json={"row":1,"col":3})
                    assert r.status_code==200, r.text
                    r=await c.post(f"/api/v1/layout/runs/{run_id}/resolve",json={"row":1,"col":2})
                    assert r.status_code==409
                    after=(await c.get(f"/api/v1/layout/runs/{run_id}")).json()
                    assert after["result"]==stored["result"]  # truth never rewrites inference
                    for i,cells in enumerate([[1,2,3,4],[2,1,4,3],[4,3,2,1],[3,4,1,2]]):
                        r=await c.post("/api/v1/layout/history",json={"rows":1,"cols":4,"cells":cells,"source":f"integration-layout-{i}"})
                        assert r.status_code==200,r.text
                    r=await c.post("/api/v1/layout/evaluation",json={"rows":1,"cols":4,"max_folds":3})
                    assert r.status_code==200,r.text
                    assert len(r.json()["folds"])==3
                    assert r.json()["discussion_improved"] is False
                    r=await c.get("/api/v1/layout/models?rows=1&cols=4")
                    assert next(m for m in r.json() if m["model"]=="position")["status"]=="ready"
                    image=Image.new("RGB",(240,240),"white")
                    draw=ImageDraw.Draw(image)
                    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",48)
                    for i in range(4):
                        draw.text((i%2*120+40,i//2*120+30),str(i+1),fill="black",font=font)
                    buffer=BytesIO();image.save(buffer,format="PNG")
                    r=await c.post("/api/v1/layout/recognize",files={"file":("grid.png",buffer.getvalue(),"image/png")},
                        data={"rows":"2","cols":"2","retain":"true","corners":"[[0,0],[239,0],[239,239],[0,239]]"})
                    assert r.status_code==200,r.text
                    assert r.json()["status"]=="needs_confirmation"
                    assert r.json()["cells"]==[1,2,3,4],r.json()
                    capture=r.json()["capture_id"]
                    r=await c.post("/api/v1/layout/ocr/corrections",json={
                        "capture_id":capture,"source_board":"integration-ocr-board",
                        "auto_train":False,"cells":[{"index":i,"state":"OPENED","label":i+1,
                                                    "scenarios":["clean"]} for i in range(4)]})
                    assert r.status_code==200,r.text
                    assert r.json()["saved"]==4 and r.json()["stage"]=="not_ready"
                    r=await c.get("/api/v1/layout/ocr/dataset")
                    assert r.status_code==200 and r.json()["labelled_cells"]>=4
                    r=await c.post("/api/v1/layout/ocr/digit/train")
                    assert r.status_code==200 and r.json()["status"]=="not_ready"

        finally:
            app.dependency_overrides.pop(get_db,None)
            await session.close()
            await transaction.rollback()
    await engine.dispose()
