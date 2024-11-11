# app/main.py
from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import asyncio
import json
from typing import Dict, Any
from datetime import datetime

from app.core.config import settings
from app.core.logging_config import logger
from app.db.database import get_db, engine, Base
from app.api import routes
from app.ml.model_updater import ModelUpdater
from app.ml.monitoring import ModelMonitor
from app.utils.cache import cache_manager
from app.utils.exceptions import handle_ml_exception, MLFrameworkException

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

# قاموس لتخزين حالات المعالجة
processing_status: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    """تهيئة المكونات عند بدء التشغيل"""
    try:
        # إنشاء جداول قاعدة البيانات
        Base.metadata.create_all(bind=engine)

        # بدء مراقبة النماذج
        if settings.MONITORING_ENABLED:
            monitor = ModelMonitor("system")
            # asyncio.create_task(monitor.start_monitoring())

        # بدء محدث النماذج
        if settings.AUTO_UPDATE_ENABLED:
            updater = ModelUpdater()
            asyncio.create_task(updater.start())

        logger.info("تم بدء تشغيل التطبيق بنجاح")
    except Exception as e:
        logger.error(f"خطأ في بدء التشغيل: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """تنظيف عند إيقاف التشغيل"""
    logger.info("جاري إيقاف التطبيق")

@app.get("/")
async def read_root(request: Request):
    """الصفحة الرئيسية"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard/{model_id}")
async def dashboard(request: Request, model_id: str, db: Session = Depends(get_db)):
    """لوحة التحكم للنموذج"""
    try:
        # التحقق من وجود النموذج
        model = await cache_manager.get_model_metadata(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="النموذج غير موجود")

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "model_id": model_id,
                "model_data": model
            }
        )
    except Exception as e:
        logger.error(f"خطأ في عرض لوحة التحكم: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{model_id}")
async def show_results(request: Request, model_id: str, db: Session = Depends(get_db)):
    """عرض نتائج التدريب"""
    try:
        # التحقق من وجود النموذج
        model = await cache_manager.get_model_metadata(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="النموذج غير موجود")

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "model_id": model_id,
                "model_data": model
            }
        )
    except Exception as e:
        logger.error(f"خطأ في عرض النتائج: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/processing/{job_id}")
async def processing_status_websocket(websocket: WebSocket, job_id: str):
    """WebSocket لتحديثات حالة المعالجة"""
    await websocket.accept()
    try:
        while True:
            if job_id in processing_status:
                await websocket.send_json(processing_status[job_id])
                if processing_status[job_id]["status"] in ["completed", "failed"]:
                    break
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"خطأ في WebSocket: {str(e)}")
    finally:
        await websocket.close()

@app.get("/api/v1/processing-status/{job_id}")
async def get_processing_status(job_id: str):
    """الحصول على حالة المعالجة"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="مهمة المعالجة غير موجودة")
    return processing_status[job_id]

def update_processing_status(job_id: str, status: str, progress: int, message: str = None):
    """تحديث حالة المعالجة"""
    processing_status[job_id] = {
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.exception_handler(MLFrameworkException)
async def ml_framework_exception_handler(request: Request, exc: MLFrameworkException):
    """معالج الأخطاء المخصص"""
    return handle_ml_exception(exc)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )