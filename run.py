import uvicorn
from app.main import app
from app.core.config import settings
import threading
from app.ml.model_updater import ModelUpdater
from app.db.database import Base, engine

def init_db():
    # حذف جميع الجداول
    Base.metadata.drop_all(bind=engine)
    # إعادة إنشاء الجداول
    Base.metadata.create_all(bind=engine)

def start_model_updater():
    updater = ModelUpdater()
    updater.start()

if __name__ == "__main__":
    # تهيئة قاعدة البيانات
    # init_db()
    
    # Start model updater in a separate thread
    updater_thread = threading.Thread(target=start_model_updater)
    updater_thread.start()

    # Run the FastAPI app
    uvicorn.run(
        "app.main:app",
        host=settings.HOST if hasattr(settings, 'HOST') else "0.0.0.0",
        port=settings.PORT if hasattr(settings, 'PORT') else 8000,
        reload=settings.DEBUG
    )