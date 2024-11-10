import uvicorn
from app.main import app
from app.core.config import settings
import threading
from app.ml.model_updater import start_model_updater

if __name__ == "__main__":
    # Start model updater in a separate thread
    updater_thread = threading.Thread(target=start_model_updater)
    updater_thread.start()

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )