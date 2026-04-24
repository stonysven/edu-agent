"""
这个文件的作用：
提供一个“轻量但可用”的并发保护与限流中间件。

为什么 AI 系统比普通后端更需要并发控制
--------------------------------------------------
普通后端接口很多时候只是：
- 查一次数据库
- 写一次缓存
- 返回一个结果

但 AI 系统的单次请求通常更重：
- 可能要调用外部 LLM
- 可能要调用 embedding
- 可能要走 RAG 检索
- 可能要经过多 Agent 编排

这意味着：
1. 单个请求耗时更长
2. 对外部依赖更敏感
3. 并发堆积时更容易形成“雪崩”

如果没有限流和并发保护，
高峰期会出现：
- 本地 worker 被占满
- 上游模型接口排队变长
- 请求整体超时率上升

所以这里做两层保护：
1. 时间窗口限流：
   限制同一个客户端在一段时间内发太多请求
2. 最大并发请求数：
   限制当前服务同时处理的活跃请求数量

为什么这里先用内存实现：
因为当前目标是教学和快速落地。
生产环境里可以进一步换成 Redis 版分布式限流。
--------------------------------------------------
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response


class SimpleRateLimitMiddleware(BaseHTTPMiddleware):
    """
    这个类的作用：
    对 FastAPI 请求增加基础的限流与并发保护。
    """

    def __init__(
        self,
        app,
        rate_limit_requests: int,
        rate_limit_window_seconds: int,
        max_concurrent_requests: int,
    ) -> None:
        super().__init__(app)
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.max_concurrent_requests = max_concurrent_requests

        # 为什么这里要同时维护锁和计数器：
        # 因为限流和并发保护都属于“共享状态”。
        # 在异步并发场景下，如果不加锁，就可能出现竞态条件。
        self._lock = asyncio.Lock()
        self._request_timestamps: dict[str, deque[float]] = defaultdict(deque)
        self._inflight_requests = 0

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """
        这个方法的作用：
        在每个请求真正进入业务逻辑前，先做基础保护检查。
        """

        client_key = self._get_client_key(request=request)
        request_accepted = False

        async with self._lock:
            now = time.time()
            timestamps = self._request_timestamps[client_key]

            # 为什么要先清理过期时间戳：
            # 因为我们用的是“滑动时间窗口”思路。
            # 过了窗口期的请求不应该继续算在限流里。
            while timestamps and now - timestamps[0] > self.rate_limit_window_seconds:
                timestamps.popleft()

            if len(timestamps) >= self.rate_limit_requests:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "请求过于频繁，请稍后再试。",
                    },
                )

            if self._inflight_requests >= self.max_concurrent_requests:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "当前并发请求过多，请稍后重试。",
                    },
                )

            timestamps.append(now)
            self._inflight_requests += 1
            request_accepted = True

        try:
            response = await call_next(request)
            return response
        finally:
            if request_accepted:
                async with self._lock:
                    self._inflight_requests = max(0, self._inflight_requests - 1)

    def _get_client_key(self, request: Request) -> str:
        """
        这个方法的作用：
        为请求生成一个基础客户端标识。

        当前这里优先使用客户端 IP。
        这是最简单直接的方案，适合教学版服务。
        """

        forwarded_for = request.headers.get("x-forwarded-for", "").strip()
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        client = request.client
        if client is None:
            return "unknown"

        return client.host
