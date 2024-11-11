# app/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.core.config import settings

# إنشاء محرك قاعدة البيانات
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}  # فقط لـ SQLite
)

# إنشاء محرك قاعدة البيانات غير المتزامن
async_engine = create_async_engine(
    settings.DATABASE_URL.replace('sqlite:///', 'sqlite+aiosqlite:///'),
    connect_args={"check_same_thread": False}
)

# إنشاء جلسة قاعدة البيانات
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# إنشاء جلسة قاعدة البيانات غير المتزامنة
async_session = sessionmaker(
    async_engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# إنشاء النموذج الأساسي
Base = declarative_base()

# دالة للحصول على جلسة قاعدة البيانات
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# دالة للحصول على جلسة قاعدة البيانات غير المتزامنة
async def get_async_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()