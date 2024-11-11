# app/utils/cache.py
from functools import lru_cache
from typing import Dict, Any, Optional, List
import redis
import json
from app.core.config import settings
from app.core.logging_config import logger

class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.initialized = False

    async def initialize(self):
        """تهيئة اتصال Redis"""
        try:
            if not self.initialized:
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=0,
                    decode_responses=True
                )
                # التحقق من الاتصال
                self.redis_client.ping()
                self.initialized = True
                logger.info("تم تهيئة ذاكرة التخزين المؤقت بنجاح")
        except redis.ConnectionError as e:
            logger.error(f"فشل الاتصال بـ Redis: {str(e)}")
            # استخدام ذاكرة مؤقتة في الذاكرة إذا فشل الاتصال بـ Redis
            self.in_memory_cache = {}
            logger.warning("تم التحويل إلى التخزين المؤقت في الذاكرة")
        except Exception as e:
            logger.error(f"خطأ في تهيئة ذاكرة التخزين المؤقت: {str(e)}")
            raise

    async def close(self):
        """إغلاق اتصال Redis"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("تم إغلاق اتصال Redis بنجاح")
        except Exception as e:
            logger.error(f"خطأ في إغلاق اتصال Redis: {str(e)}")

    async def get_cached_prediction(self, input_hash: str) -> Optional[Dict[str, Any]]:
        """الحصول على نتيجة التنبؤ المخزنة مؤقتاً"""
        try:
            if self.redis_client:
                cached = await self.redis_client.get(f"pred:{input_hash}")
                if cached:
                    return json.loads(cached)
            elif hasattr(self, 'in_memory_cache'):
                return self.in_memory_cache.get(f"pred:{input_hash}")
            return None
        except Exception as e:
            logger.error(f"خطأ في استرجاع التنبؤ المخزن: {str(e)}")
            return None

    async def cache_prediction(self, input_hash: str, result: Dict[str, Any], ttl: int = 3600):
        """تخزين نتيجة التنبؤ مؤقتاً"""
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    f"pred:{input_hash}",
                    ttl,
                    json.dumps(result)
                )
            elif hasattr(self, 'in_memory_cache'):
                self.in_memory_cache[f"pred:{input_hash}"] = result
        except Exception as e:
            logger.error(f"خطأ في تخزين التنبؤ: {str(e)}")

    async def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """الحصول على البيانات الوصفية للنموذج"""
        try:
            if self.redis_client:
                metadata = await self.redis_client.get(f"model:{model_id}")
                return json.loads(metadata) if metadata else None
            elif hasattr(self, 'in_memory_cache'):
                return self.in_memory_cache.get(f"model:{model_id}")
            return None
        except Exception as e:
            logger.error(f"خطأ في استرجاع بيانات النموذج: {str(e)}")
            return None

    async def cache_model_metadata(self, model_id: str, metadata: Dict[str, Any], ttl: int = 86400):
        """تخزين البيانات الوصفية للنموذج"""
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    f"model:{model_id}",
                    ttl,
                    json.dumps(metadata)
                )
            elif hasattr(self, 'in_memory_cache'):
                self.in_memory_cache[f"model:{model_id}"] = metadata
        except Exception as e:
            logger.error(f"خطأ في تخزين بيانات النموذج: {str(e)}")

    async def get_all_models(self) -> List[Dict[str, Any]]:
        """الحصول على قائمة جميع النماذج"""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys("model:*")
                models = []
                for key in keys:
                    model_data = await self.redis_client.get(key)
                    if model_data:
                        models.append(json.loads(model_data))
                return models
            elif hasattr(self, 'in_memory_cache'):
                return [
                    value for key, value in self.in_memory_cache.items()
                    if key.startswith("model:")
                ]
            return []
        except Exception as e:
            logger.error(f"خطأ في استرجاع قائمة النماذج: {str(e)}")
            return []

    async def clear_cache(self):
        """مسح جميع البيانات المخزنة مؤقتاً"""
        try:
            if self.redis_client:
                await self.redis_client.flushdb()
            elif hasattr(self, 'in_memory_cache'):
                self.in_memory_cache.clear()
            logger.info("تم مسح ذاكرة التخزين المؤقت")
        except Exception as e:
            logger.error(f"خطأ في مسح ذاكرة التخزين المؤقت: {str(e)}")

cache_manager = CacheManager()