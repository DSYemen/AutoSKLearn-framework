# app/utils/cache.py
from functools import lru_cache
from typing import Dict, Any
import redis
from app.core.config import settings

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0
        )

    def get_cached_prediction(self, input_hash: str) -> Dict[str, Any]:
        """Get cached prediction result"""
        cached = self.redis_client.get(f"pred:{input_hash}")
        return cached.decode() if cached else None

    def cache_prediction(self, input_hash: str, result: Dict[str, Any], ttl: int = 3600):
        """Cache prediction result"""
        self.redis_client.setex(
            f"pred:{input_hash}",
            ttl,
            json.dumps(result)
        )

    @lru_cache(maxsize=1000)
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get cached model metadata"""
        return self.redis_client.get(f"model:{model_id}")

cache_manager = CacheManager()