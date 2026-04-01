"""
这个文件的作用：
实现当前项目的 Memory 系统，也就是“多轮对话记忆管理器”。

为什么需要 Memory：
大语言模型（LLM）本身是“无状态”的。
这句话的意思是：
模型不会天然记住你上一轮说过什么，
每次调用它时，它看到的只有你这一次请求里显式传给它的内容。

这会带来一个直接问题：
如果我们不自己保存历史消息，那么：
- 用户上一轮说过的话会丢失
- 模型无法理解上下文中的“它”“刚才那个问题”“继续展开”等指代
- 多轮对话体验会很差

所以，Memory 系统的核心职责就是：
1. 记录某个会话（session）里的历史消息
2. 在下次请求时，把合适的历史再交给 Agent

概念解释注释块：为什么不能无限拼历史
--------------------------------------------------
虽然多轮对话需要历史，但历史不能无限增长。

原因主要有 3 个：
1. Token 成本会不断升高，调用越来越贵
2. 请求体会越来越长，响应速度会变慢
3. 太长的历史里会混入很多旧信息，反而可能干扰当前问题

所以工程上常见的做法不是“把所有历史都传给模型”，
而是做“上下文控制”。

当前这个教学版项目采用最简单、最稳妥的策略：
- 默认只取最近 5 轮对话

这里的“5 轮”在实现上表现为“最近 10 条消息”，也就是：
- 5 条 user
- 5 条 assistant
--------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Protocol

try:
    import redis
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover
    redis = None
    Redis = None

    class RedisError(Exception):
        """在没有安装 redis 包时，提供一个占位异常类型。"""


Message = dict[str, str]


class MemoryStore(Protocol):
    """约束不同存储实现需要提供相同的方法。"""

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """向指定会话追加一条消息。"""

    def get_history(self, session_id: str) -> list[Message]:
        """获取指定会话的全部历史。"""

    def clear_history(self, session_id: str) -> None:
        """清空指定会话的历史。"""


class InMemoryStore:
    """
    这个类的作用：
    提供一个纯内存版的消息存储实现。

    为什么需要它：
    这样即使没有 Redis，项目也仍然可以直接运行。
    """

    def __init__(self) -> None:
        self._store: dict[str, list[Message]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> list[Message]:
        return list(self._store.get(session_id, []))

    def clear_history(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class RedisMemoryStore:
    """
    这个类的作用：
    提供基于 Redis 的消息存储实现。
    """

    def __init__(self, redis_client: Redis, key_prefix: str = "edu-agent:memory") -> None:
        self.redis_client = redis_client
        self.key_prefix = key_prefix

    def _build_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}"

    def add_message(self, session_id: str, role: str, content: str) -> None:
        message = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        self.redis_client.rpush(self._build_key(session_id), message)

    def get_history(self, session_id: str) -> list[Message]:
        raw_messages = self.redis_client.lrange(self._build_key(session_id), 0, -1)
        history: list[Message] = []
        for raw_message in raw_messages:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            history.append(json.loads(raw_message))
        return history

    def clear_history(self, session_id: str) -> None:
        self.redis_client.delete(self._build_key(session_id))


class MemoryManager:
    """
    这个类的作用：
    作为上层统一可用的 Memory 管理入口。
    """

    def __init__(self, redis_url: str = "", default_history_limit: int = 5) -> None:
        self.default_history_limit = default_history_limit
        self.store = self._build_store(redis_url=redis_url)

    def _build_store(self, redis_url: str) -> MemoryStore:
        """
        这个方法的作用：
        按“Redis 优先、内存兜底”的策略选择底层存储。
        """

        if redis_url and redis is not None:
            try:
                redis_client = redis.from_url(redis_url, decode_responses=False)
                redis_client.ping()
                return RedisMemoryStore(redis_client=redis_client)
            except RedisError:
                return InMemoryStore()

        return InMemoryStore()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.store.add_message(session_id=session_id, role=role, content=content)

    def get_history(self, session_id: str) -> list[Message]:
        return self.store.get_history(session_id=session_id)

    def get_recent_history(self, session_id: str, limit: int | None = None) -> list[Message]:
        """
        这个方法的作用：
        获取最近几轮对话历史。

        为什么这里的 `limit` 表示“轮数”而不是“消息条数”：
        因为从业务视角看，开发者更容易理解“最近 5 轮对话”。

        具体实现方式：
        - 1 轮对话通常包含 2 条消息：user + assistant
        - 所以 `limit=5` 时，我们会返回最近 10 条消息
        """

        history = self.get_history(session_id=session_id)
        round_limit = limit or self.default_history_limit
        message_limit = round_limit * 2
        return history[-message_limit:]

    def clear_history(self, session_id: str) -> None:
        self.store.clear_history(session_id=session_id)
