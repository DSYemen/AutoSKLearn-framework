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

from app.core.config import settings
from app.core.logging_config import logger
from app.db.database import get_db, engine, Base
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

# قاموس لتخزين اتصالات WebSocket
websocket_connections: Dict[str, List[WebSocket]] = {}

# قاموس لتخزين حالات المعالجة
processing_status: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    """تهيئة المكونات عند بدء التشغيل"""
    try:
        # إنشاء جداول قاعدة البيانات
        Base.metadata.create_all(bind=engine)

        # تهيئة الذاكرة المؤقتة
        await cache_manager.initialize()

        # بدء مراقبة النماذج
        if settings.MONITORING_ENABLED:
            monitor = ModelMonitor("system")
            asyncio.create_task(monitor.start_monitoring())

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
    try:
        # إغلاق اتصالات WebSocket
        for connections in websocket_connections.values():
            for websocket in connections:
                await websocket.close()

        # إغلاق الذاكرة المؤقتة
        await cache_manager.close()

        logger.info("تم إيقاف التطبيق بنجاح")
    except Exception as e:
        logger.error(f"خطأ في إيقاف التشغيل: {str(e)}")

@app.get("/")
async def read_root(request: Request, db: Session = Depends(get_db)):
    """الصفحة الرئيسية"""
    try:
        # الحصول على إحصائيات النظام
        stats = {
            "active_models": await get_active_models_count(db),
            "total_predictions": await get_total_predictions_count(db),
            "avg_accuracy": await get_average_accuracy(db),
            "system_health": 100,  # يمكن تحديثها بناءً على حالة النظام
            "predictions_per_hour": await get_predictions_per_hour(db),
            "health_status": "Healthy"
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
        # إرجاع إحصائيات افتراضية في حالة الخطأ
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "stats": {
                    "active_models": 0,
                    "total_predictions": 0,
                    "avg_accuracy": 0,
                    "system_health": 0,
                    "predictions_per_hour": 0,
                    "health_status": "Error"
                }
            }
        )

async def get_active_models_count(db: Session) -> int:
    """الحصول على عدد النماذج النشطة"""
    return db.query(ModelRecord).filter(ModelRecord.status == "active").count()

async def get_total_predictions_count(db: Session) -> int:
    """الحصول على إجمالي عدد التنبؤات"""
    return db.query(PredictionLog).count()

async def get_average_accuracy(db: Session) -> float:
    """الحصول على متوسط الدقة"""
    models = db.query(ModelRecord).all()
    if not models:
        return 0.0
    accuracies = [model.metrics.get("accuracy", 0) for model in models]
    return sum(accuracies) / len(accuracies)

async def get_predictions_per_hour(db: Session) -> float:
    """الحصول على معدل التنبؤات في الساعة"""
    hour_ago = datetime.utcnow() - timedelta(hours=1)
    return db.query(PredictionLog).filter(PredictionLog.timestamp >= hour_ago).count()

@app.get("/dashboard/{model_id}")
async def dashboard(request: Request, model_id: str, db: Session = Depends(get_db)):
    """لوحة التحكم للنموذج"""
    try:
        # التحقق من وجود النموذج
        model = await cache_manager.get_model_metadata(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="النموذج غير موجود")

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
    try:
        await websocket.accept()
        
        # إضافة الاتصال إلى القاموس
        if job_id not in websocket_connections:
            websocket_connections[job_id] = []
        websocket_connections[job_id].append(websocket)

        try:
            while True:
                if job_id in processing_status:
                    await websocket.send_json(processing_status[job_id])
                    if processing_status[job_id]["status"] in ["completed", "failed"]:
                        break
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            websocket_connections[job_id].remove(websocket)
            if not websocket_connections[job_id]:
                del websocket_connections[job_id]
    except Exception as e:
        logger.error(f"خطأ في WebSocket: {str(e)}")
    finally:
        if websocket in websocket_connections.get(job_id, []):
            websocket_connections[job_id].remove(websocket)

@app.get("/api/v1/processing-status/{job_id}")
async def get_processing_status(job_id: str):
    """الحصول على حالة المعالجة"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="مهمة المعالجة غير موجودة")
    return processing_status[job_id]

async def broadcast_status_update(job_id: str, status: Dict[str, Any]):
    """بث تحديث الحالة لجميع العملاء المتصلين"""
    if job_id in websocket_connections:
        for websocket in websocket_connections[job_id]:
            try:
                await websocket.send_json(status)
            except Exception as e:
                logger.error(f"خطأ في بث التحديث: {str(e)}")

def update_processing_status(job_id: str, status: str, progress: int, message: str = None):
    """تحديث حالة المعالجة"""
    status_update = {
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    processing_status[job_id] = status_update
    
    # بث التحديث عبر WebSocket
    asyncio.create_task(broadcast_status_update(job_id, status_update))

@app.exception_handler(MLFrameworkException)
async def ml_framework_exception_handler(request: Request, exc: MLFrameworkException):
    """معالج الأخطاء المخصص"""
    return handle_ml_exception(exc)

@app.get("/models")
async def list_models(request: Request, db: Session = Depends(get_db)):
    """صفحة قائمة النماذج"""
    models = await cache_manager.get_all_models()
    return templates.TemplateResponse(
        "models.html",
        {
            "request": request,
            "models": models
        }
    )

@app.get("/predictions")
async def predictions_page(request: Request, db: Session = Depends(get_db)):
    """صفحة التنبؤات"""
    return templates.TemplateResponse(
        "predictions.html",
        {
            "request": request
        }
    )

@app.get("/data/profile")
async def data_profile_page(request: Request):
    """صفحة تحليل البيانات"""
    return templates.TemplateResponse(
        "profile_report.html",
        {
            "request": request
        }
    )

@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # إرسال تحديثات كل ثانية
            await asyncio.sleep(1)
            stats = await get_system_stats()
            await websocket.send_json({
                "type": "stats_update",
                "stats": stats
            })
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST if hasattr(settings, 'HOST') else "0.0.0.0",
        port=settings.PORT if hasattr(settings, 'PORT') else 8000,
        reload=settings.DEBUG
    )