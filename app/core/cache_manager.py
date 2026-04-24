"""
这个文件的作用：
提供当前项目统一可用的缓存管理器。

为什么 AI 系统非常适合做缓存
--------------------------------------------------
AI 系统里有两类调用特别适合缓存：

1. LLM 回答：
   如果输入 prompt 完全一样，
   那么重复调用模型通常只是在重复花钱和重复等待。

2. embedding：
   同一段文本做 embedding，结果通常是稳定的。
   所以它是典型的“高价值缓存对象”。

缓存对成本的影响为什么很大：
1. LLM 调用通常按 token 计费，命中缓存就等于直接省掉一次完整费用
2. embedding 虽然单次可能比聊天便宜，但批量建索引时数量很多，累计成本不小
3. 命中缓存后，本地读取速度远快于重新访问模型服务

所以缓存带来的收益通常有两层：
- 降低模型调用成本
- 降低整体响应延迟

为什么这里既支持 Redis 又支持内存：
因为你要求“Redis 或内存实现缓存”。
所以这里采用和 Memory 系统一样的思路：
- 优先尝试 Redis
- 失败时回退到内存

这样项目在没有 Redis 的本地环境里也能直接运行。
--------------------------------------------------
"""

from __future__ import annotations

import asyncio
import json
import time
from functools import lru_cache
from hashlib import sha256
from typing import Any, Protocol

try:
    import redis.asyncio as redis_async
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover
    redis_async = None

    class RedisError(Exception):
        """在没有安装 redis 包时，提供一个占位异常类型。"""


from app.core.config import Settings, get_settings


class CacheStore(Protocol):
    """约束不同缓存实现暴露统一的异步接口。"""

    async def get(self, key: str) -> str | None:
        """读取缓存值。"""

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """写入缓存值。"""


class InMemoryCacheStore:
    """
    这个类的作用：
    提供一个纯内存版缓存实现。

    为什么要做过期时间：
    因为缓存不是永久真理。
    如果不设 TTL，旧结果会一直堆积，既占内存，也可能逐渐过时。
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            cached_item = self._store.get(key)
            if cached_item is None:
                return None

            expires_at, value = cached_item
            if expires_at < time.time():
                self._store.pop(key, None)
                return None

            return value

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        expires_at = time.time() + ttl_seconds
        async with self._lock:
            self._store[key] = (expires_at, value)


class RedisCacheStore:
    """
    这个类的作用：
    提供基于 Redis 的缓存实现。
    """

    def __init__(self, redis_client: Any) -> None:
        self.redis_client = redis_client

    async def get(self, key: str) -> str | None:
        value = await self.redis_client.get(key)
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        await self.redis_client.set(key, value, ex=ttl_seconds)


class CacheManager:
    """
    这个类的作用：
    统一管理缓存读写和 hit/miss 统计。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = None
        self.backend = "memory"
        self._stats_lock = asyncio.Lock()
        self._hit_count = 0
        self._miss_count = 0
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def ensure_initialized(self) -> None:
        """
        这个方法的作用：
        在真正使用缓存前，按需初始化底层缓存实现。
        """

        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            self.store = await self._build_store()
            self._initialized = True

    async def _build_store(self) -> CacheStore:
        """
        这个方法的作用：
        优先尝试 Redis，失败时回退到内存缓存。
        """

        if self.settings.redis_url and redis_async is not None:
            try:
                redis_client = redis_async.from_url(
                    self.settings.redis_url,
                    decode_responses=False,
                )
                await redis_client.ping()
                self.backend = "redis"
                return RedisCacheStore(redis_client=redis_client)
            except RedisError:
                self.backend = "memory"
                return InMemoryCacheStore()

        self.backend = "memory"
        return InMemoryCacheStore()

    def build_cache_key(self, namespace: str, payload: dict[str, Any]) -> str:
        """
        这个方法的作用：
        根据命名空间和结构化 payload 生成稳定缓存键。

        为什么这里不用“直接拿用户问题做 key”：
        因为 AI 系统里的同一句问题，可能对应不同上下文。
        比如：
        - 有历史上下文时
        - 没历史上下文时
        - 不同模型时

        如果只看用户问题，很容易错误命中缓存。
        所以这里使用“完整语义输入”的哈希作为 key。
        """

        normalized_payload = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = sha256(normalized_payload.encode("utf-8")).hexdigest()
        return f"{self.settings.cache_prefix}:{namespace}:{digest}"

    async def get_json(self, namespace: str, payload: dict[str, Any]) -> Any | None:
        """
        这个方法的作用：
        从缓存中读取 JSON 数据，并自动记录 hit/miss。
        """

        await self.ensure_initialized()
        cache_key = self.build_cache_key(namespace=namespace, payload=payload)
        cached_value = await self.store.get(cache_key)

        if cached_value is None:
            await self.record_miss()
            return None

        await self.record_hit()
        return json.loads(cached_value)

    async def set_json(
        self,
        namespace: str,
        payload: dict[str, Any],
        value: Any,
        ttl_seconds: int,
    ) -> None:
        """
        这个方法的作用：
        把 JSON 数据写入缓存。
        """

        await self.ensure_initialized()
        cache_key = self.build_cache_key(namespace=namespace, payload=payload)
        serialized_value = json.dumps(value, ensure_ascii=False)
        await self.store.set(cache_key, serialized_value, ttl_seconds=ttl_seconds)

    async def record_hit(self) -> None:
        async with self._stats_lock:
            self._hit_count += 1

    async def record_miss(self) -> None:
        async with self._stats_lock:
            self._miss_count += 1

    async def get_stats(self) -> dict[str, int | str]:
        """
        这个方法的作用：
        返回当前缓存命中统计。
        """

        await self.ensure_initialized()
        async with self._stats_lock:
            return {
                "backend": self.backend,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
            }


@lru_cache
def get_cache_manager() -> CacheManager:
    """
    这个函数的作用：
    返回一个全局复用的缓存管理器实例。
    """

    return CacheManager()
