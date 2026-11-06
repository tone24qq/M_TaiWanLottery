"""Aggregate API v1 router."""

from fastapi import APIRouter

from taiwan_lottery.api.v1.endpoints import (
    lotto649,
    super_lotto,
    daily_cash,
    bingo,
    scraper,
    statistics,
    ml,
)

api_router = APIRouter()

api_router.include_router(lotto649.router, prefix="/lotto649", tags=["大樂透"])
api_router.include_router(super_lotto.router, prefix="/super_lotto", tags=["威力彩"])
api_router.include_router(daily_cash.router, prefix="/daily_cash", tags=["今彩539"])
api_router.include_router(bingo.router, prefix="/bingo", tags=["賓果賓果"])
api_router.include_router(scraper.router, prefix="/scraper", tags=["爬蟲"])
api_router.include_router(statistics.router, prefix="/stats", tags=["統計分析"])
api_router.include_router(ml.router, prefix="/ml", tags=["ML 預測"])

from taiwan_lottery.api.v1.endpoints import layout_prediction
api_router.include_router(layout_prediction.router, prefix="/layout", tags=["表格 AI 討論"])
