# app/main.py
from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import psutil

from app.core.config import settings
from app.core.logging_config import logger
from app.db.database import get_db, engine, Base, async_session
from app.api import routes, docs
from app.ml.model_updater import ModelUpdater
from app.ml.monitoring import ModelMonitor
from app.utils.cache import cache_manager
from app.utils.exceptions import handle_ml_exception, MLFrameworkException
from app.visualization.dashboard import DashboardGenerator

# إنشاء التطبيق
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# تهيئة القوالب
templates = Jinja2Templates(directory="templates")

# إضافة الملفات الثابتة
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تضمين المسارات
app.include_router(
    routes.router,
    prefix=settings.API_PREFIX,
    tags=["ml"]
)
app.include_router(
    docs.docs_router,
    prefix=settings.API_PREFIX,
    tags=["docs"]
)

@app.get("/")
async def read_root(request: Request, db: Session = Depends(get_db)):
    """الصفحة الرئيسية"""
    try:
        stats = {
            "active_models": await get_active_models_count(db),
            "total_predictions": await get_total_predictions_count(db),
            "avg_accuracy": await get_average_accuracy(db),
            "system_health": 100,
            "predictions_per_hour": await get_predictions_per_hour(db)
        }
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "stats": stats
            }
        )
    except Exception as e:
        logger.error(f"Error in read_root: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "stats": {
                    "active_models": 0,
                    "total_predictions": 0,
                    "avg_accuracy": 0,
                    "system_health": 100,
                    "predictions_per_hour": 0
                }
            }
        )

@app.get("/dashboard/{model_id}")
async def show_dashboard(request: Request, model_id: str, db: Session = Depends(get_db)):
    """عرض لوحة التحكم للنموذج"""
    try:
        # الحصول على بيانات النموذج
        model = await cache_manager.get_model_metadata(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # إنشاء لوحة التحكم
        dashboard_generator = DashboardGenerator()
        dashboard_data = await dashboard_generator.generate_dashboard_data(model_id, db)

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "model_id": model_id,
                "model_data": model,
                "dashboard_data": dashboard_data
            }
        )
    except Exception as e:
        logger.error(f"Error showing dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models(request: Request, db: Session = Depends(get_db)):
    """صفحة قائمة النماذج"""
    try:
        models = db.query(ModelRecord).all()
        return templates.TemplateResponse(
            "models.html",
            {
                "request": request,
                "models": models
            }
        )
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return templates.TemplateResponse(
            "models.html",
            {
                "request": request,
                "models": []
            }
        )

@app.get("/data/profile")
async def data_profile_page(request: Request, db: Session = Depends(get_db)):
    """صفحة تحليل البيانات"""
    try:
        profiles = db.query(DatasetProfile).order_by(
            DatasetProfile.created_at.desc()
        ).all()
        return templates.TemplateResponse(
            "profile_report.html",
            {
                "request": request,
                "profiles": profiles
            }
        )
    except Exception as e:
        logger.error(f"Error showing data profiles: {str(e)}")
        return templates.TemplateResponse(
            "profile_report.html",
            {
                "request": request,
                "profiles": []
            }
        )

@app.get("/api/docs")
async def api_docs(request: Request):
    """صفحة توثيق API"""
    return templates.TemplateResponse(
        "api_docs.html",
        {
            "request": request
        }
    )

# باقي الكود كما هو...