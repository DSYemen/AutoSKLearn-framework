# app/main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging_config import logger
from app.db.database import get_db
from app.api import routes
from app.ml.model_updater import ModelUpdater
import threading

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    routes.router,
    prefix=settings.API_PREFIX,
    tags=["ml"]
)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        # Initialize database
        from app.db.database import Base, engine
        Base.metadata.create_all(bind=engine)

        # Start model updater in background
        if settings.MONITORING_ENABLED:
            updater = ModelUpdater()
            threading.Thread(target=updater.start, daemon=True).start()

        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Application shutting down")