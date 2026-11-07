"""FastAPI application entry point."""

import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from taiwan_lottery.config import settings

# Windows asyncio policy for asyncpg compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure loguru
logger.remove()
logger.add(sys.stderr, level="DEBUG" if settings.DEBUG else "INFO")
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    logger.info("Starting {} ...", settings.APP_NAME)

    # Ensure model artifacts directory exists
    settings.MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Start scheduler if enabled
    if settings.SCRAPER_ENABLED:
        try:
            from taiwan_lottery.scraper.scheduler import start_scheduler
            start_scheduler()
            logger.info("Scraper scheduler started")
        except Exception as e:
            logger.warning("Failed to start scheduler: {}", e)

    yield

    # Shutdown
    if settings.SCRAPER_ENABLED:
        try:
            from taiwan_lottery.scraper.scheduler import stop_scheduler
            stop_scheduler()
        except Exception:
            pass

    from taiwan_lottery.db.engine import engine
    await engine.dispose()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="台灣彩券數據收集與 ML 預測系統",
    lifespan=lifespan,
)

# Static files and templates
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Include API routers
from taiwan_lottery.api.v1.router import api_router  # noqa: E402
app.include_router(api_router, prefix="/api/v1")


# --- Page routes (Jinja2) ---

@app.get("/", include_in_schema=False)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/history/{game}", include_in_schema=False)
async def history_page(request: Request, game: str):
    return templates.TemplateResponse("history.html", {"request": request, "game": game})


@app.get("/analysis/{game}", include_in_schema=False)
async def analysis_page(request: Request, game: str):
    return templates.TemplateResponse("analysis.html", {"request": request, "game": game})


@app.get("/predictions", include_in_schema=False)
async def predictions_page(request: Request):
    return templates.TemplateResponse("predictions.html", {"request": request})


@app.get("/bingo", include_in_schema=False)
async def bingo_page(request: Request):
    return templates.TemplateResponse("bingo.html", {"request": request})


@app.get("/layout", include_in_schema=False)
async def layout_page(request: Request):
    return templates.TemplateResponse(request=request, name="layout.html")
