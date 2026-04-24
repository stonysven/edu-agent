"""
这个文件的作用：
实现项目中的第一个“可用 Agent”，并让它支持多轮对话。

为什么叫 `simple_agent`：
因为它依然是一个最小版本的 Agent，
但现在比第一版多了一个重要能力：
- 可以读取并使用历史对话

这意味着它已经具备了最基础的“上下文连续性”。

概念解释注释块：为什么 token 优化很重要
--------------------------------------------------
在 Agent 系统里，token 优化不是“可有可无的小优化”，
而是会直接影响成本、性能和稳定性的核心工程问题。

为什么这么说：
1. 成本：
   大模型接口通常按 token 计费。
   prompt 越长，每次请求越贵；并发越高，总成本越明显。

2. 性能：
   prompt 越长，网络传输、服务端处理和模型生成都会更慢。
   对话系统里，用户通常对首字延迟非常敏感。

3. 稳定性：
   无关上下文越多，模型越容易被旧信息干扰。
   所以“不是历史越多越好”，而是“历史越相关越好”。

因此这一版优化的重点不是盲目删上下文，
而是：
- 去掉重复历史
- 缩短系统提示词
- 用可观测的 trace 看到 prompt 到底有多大
--------------------------------------------------

概念解释注释块：Memory 的意义
--------------------------------------------------
LLM 本身是无状态的。
这意味着模型不会自动记住“上一轮你说了什么”。

所以，多轮对话一定要由我们的系统自己维护历史记录，
然后在新请求到来时，把“最近相关上下文”再次传给模型。

这里的 `session_id` 就是关键：
- 它用于区分不同用户或不同会话线程
- 同一个 `session_id` 下的消息会被归到同一段历史中

为什么不能无限把所有历史都传给模型：
1. 历史越长，token 越贵
2. 请求越长，速度越慢
3. 太旧的信息可能反而干扰当前问题

因此当前版本采用一个简单策略：
- 默认只取最近 5 轮对话

这个策略不是最终形态，但非常适合教学版和第一版实现。
--------------------------------------------------

概念解释注释块：trace 的意义
--------------------------------------------------
trace 可以理解为“Agent 执行过程的可观察记录”。

为什么 AI Agent 系统需要 trace：
1. 方便调试：知道模型前后做了什么
2. 方便教学：让初学者看到 Agent 的工作流程
3. 方便扩展：以后接入工具调用、记忆、RAG 时，trace 会更有价值

这次比上一版新增了两个关键 trace：
- history_used：本次实际使用了多少条历史消息
- prompt_length：本次 prompt 的简单长度估算

这样我们就能更直观看到：
- Memory 是否真的工作了
- 上下文是否正在变长
--------------------------------------------------

概念解释注释块：LLM 调用流程
--------------------------------------------------
一次最基础的多轮对话调用，通常会经历这些步骤：

1. 根据 `session_id` 读取最近历史
2. 拼出完整 messages：
   system + history + current user
3. 通过 HTTP POST 请求发送给模型服务
4. 解析响应 JSON
5. 拿到最终文本答案
6. 把本轮 user / assistant 消息写回 Memory

为什么这里改用 `httpx.AsyncClient`：
因为当前目标已经从“先跑通”升级为“更好地承载并发请求”。
如果还继续使用同步 HTTP 请求，那么一个请求在等待 LLM 返回时，
当前 Python 协程就会一直阻塞，服务吞吐会明显受影响。
--------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import httpx

from app.agents.base_agent import AgentResult, BaseAgent
from app.core.cache_manager import CacheManager, get_cache_manager
from app.core.config import Settings, get_settings
from app.memory.memory_manager import MemoryManager


class SimpleAgent(BaseAgent):
    """
    这个类的作用：
    实现最简单的单 Agent 多轮对话能力。

    它当前的职责非常清晰：
    1. 根据 `session_id` 读取最近历史
    2. 调用 LLM 生成回答
    3. 把本轮对话写回 Memory
    4. 构造 trace 并返回统一结果
    """

    def __init__(
        self,
        settings: Settings | None = None,
        memory_manager: MemoryManager | None = None,
        cache_manager: CacheManager | None = None,
    ) -> None:
        """
        为什么允许外部传入 `settings` 和 `memory_manager`：
        这样后续做测试、替换配置或替换存储实现时会更灵活。
        """

        self.settings = settings or get_settings()
        self.memory_manager = memory_manager or MemoryManager(
            redis_url=self.settings.redis_url,
            default_history_limit=self.settings.memory_history_limit,
        )
        self.cache_manager = cache_manager or get_cache_manager()
        self.agent_name = "simple_agent"

    async def run(self, user_message: str, session_id: str) -> AgentResult:
        """
        这个方法的作用：
        执行一次支持多轮上下文的 Agent 对话流程。

        这里的执行流程分为 6 步：
        1. 根据 `session_id` 读取最近历史
        2. 构造 thought
        3. 拼接 messages
        4. 调用 LLM
        5. 写回 Memory
        6. 返回结果和 trace
        """

        trace: list[dict[str, str]] = []
        recent_history = self.memory_manager.get_recent_history(session_id=session_id)

        # 为什么这里要做一次历史优化：
        # 因为 Memory 存的是“原始对话记录”，它的职责是尽量完整保存。
        # 但真正发给模型的上下文，应该更关注“相关且不重复”。
        # 所以我们在发送前做一次轻量清洗，而不是改动底层存储。
        optimized_history = self._optimize_recent_history(
            recent_history=recent_history,
            user_message=user_message,
        )

        thought_content = self._build_thought(
            user_message=user_message,
            history_count=len(optimized_history),
        )
        trace.append({"step": "thought", "content": thought_content})

        trace.append(
            {
                "step": "history_used",
                "content": str(len(optimized_history)),
            }
        )

        trace.append(
            {
                "step": "llm_call",
                "content": f"调用 LLM 生成回答，模型为 {self.settings.openai_model}",
            }
        )

        messages = self._build_messages(
            recent_history=optimized_history,
            user_message=user_message,
        )

        prompt_length = self._estimate_messages_token_count(messages=messages)
        history_token_count = self._estimate_messages_token_count(
            messages=optimized_history,
        )
        history_ratio = self._estimate_history_ratio(
            history_token_count=history_token_count,
            total_prompt_token_count=prompt_length,
        )

        trace.append(
            {
                "step": "prompt_length",
                "content": str(prompt_length),
            }
        )
        trace.append(
            {
                "step": "history_ratio",
                "content": f"{history_ratio:.2%}",
            }
        )

        answer, cache_status = await self._call_llm(messages=messages)
        trace.append({"step": "llm_cache", "content": cache_status})

        # 只有在模型调用成功后，才把本轮消息写回 Memory。
        self.memory_manager.add_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )
        self.memory_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        trace.append({"step": "final", "content": answer})

        return AgentResult(
            answer=answer,
            agent=self.agent_name,
            session_id=session_id,
            intent="chat",
            trace=trace,
            sources=[],
        )

    def _build_thought(self, user_message: str, history_count: int) -> str:
        """
        这个方法的作用：
        构造一个简单的“思考描述”，帮助我们理解本次请求的上下文情况。
        """

        return (
            f"用户正在咨询：{user_message}。"
            f"本次回答会参考最近 {history_count} 条历史消息。"
        )

    def _optimize_recent_history(
        self,
        recent_history: list[dict[str, str]],
        user_message: str,
    ) -> list[dict[str, str]]:
        """
        这个方法的作用：
        对最近历史做一次轻量优化，减少重复上下文。

        为什么这里不做复杂摘要：
        因为当前目标是“先用简单方式稳定省 token”，
        而不是立刻引入摘要模型或更复杂的记忆压缩流程。

        当前做的事情有三类：
        1. 跳过空消息
        2. 去掉相邻重复消息
        3. 如果历史里的最后一条 user 消息和当前问题完全一样，就不再重复传入
        """

        optimized_history: list[dict[str, str]] = []

        for message in recent_history:
            role = message.get("role", "").strip()
            content = message.get("content", "").strip()

            if not role or not content:
                continue

            if optimized_history:
                previous_message = optimized_history[-1]
                if (
                    previous_message["role"] == role
                    and previous_message["content"] == content
                ):
                    continue

            optimized_history.append({"role": role, "content": content})

        # 为什么要检查“最后一条历史 user 消息”和当前输入是否重复：
        # 某些前端或上层流程可能会在重试场景里把当前问题先写进历史，
        # 如果这里不防重，模型就会看到两份一模一样的用户问题。
        if optimized_history:
            last_message = optimized_history[-1]
            if (
                last_message["role"] == "user"
                and last_message["content"].strip() == user_message.strip()
            ):
                optimized_history.pop()

        return optimized_history

    def _build_messages(
        self,
        recent_history: list[dict[str, str]],
        user_message: str,
    ) -> list[dict[str, str]]:
        """
        这个方法的作用：
        把“系统提示词 + 历史消息 + 当前用户消息”拼成最终发给 LLM 的消息列表。
        """

        # 为什么这里要精简系统提示词：
        # 因为系统提示词每次请求都会重复发送，
        # 它是最稳定、最隐性的 token 消耗来源之一。
        # 保留关键约束即可，过长说明不一定带来更好效果。
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "你是教学型AI助手。"
                    "回答要准确、简洁、连贯；"
                    "优先利用当前对话上下文，不确定时直接说明。"
                ),
            }
        ]

        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _estimate_token_count(self, text: str) -> int:
        """
        这个方法的作用：
        对文本 token 数量做一个粗略估算。

        为什么这里不用精确 tokenizer：
        因为当前项目希望继续保持依赖简单，
        不为了“统计 token”专门引入新的分词库。

        这里采用一个工程上常见的近似思路：
        - 中文通常可以近似按“每个字符接近 1 个 token”理解
        - 英文和空格混合文本通常会比字符数更少

        所以这里用“字符数 / 1.5”的方式做保守估算，
        目的不是精确计费，而是帮助我们观察趋势：
        prompt 是在变长还是变短。
        """

        normalized_text = text.strip()
        if not normalized_text:
            return 0
        return max(1, int(len(normalized_text) / 1.5))

    def _estimate_messages_token_count(self, messages: list[dict[str, str]]) -> int:
        """
        这个方法的作用：
        统计整组 messages 的估算 token 数量。
        """

        total_token_count = 0
        for message in messages:
            total_token_count += self._estimate_token_count(message.get("role", ""))
            total_token_count += self._estimate_token_count(message.get("content", ""))
        return total_token_count

    def _estimate_history_ratio(
        self,
        history_token_count: int,
        total_prompt_token_count: int,
    ) -> float:
        """
        这个方法的作用：
        估算“历史上下文”在整个 prompt 中的占比。

        为什么要看这个指标：
        因为它能帮助我们判断：
        - 当前 token 主要花在了哪里
        - 是用户当前问题占得更多，还是历史占得更多
        - 历史是否已经开始挤压当前问题空间
        """

        if total_prompt_token_count <= 0:
            return 0.0
        return history_token_count / total_prompt_token_count

    async def _call_llm(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        """
        这个方法的作用：
        通过 HTTP 请求调用 LLM，并提取最终文本答案。

        为什么这里必须做超时控制：
        因为 LLM 调用是整个 AI 服务里最慢、最不稳定的一段。
        一旦上游模型服务抖动，或者网络卡住，
        如果本地没有超时保护，请求就会长时间挂起，
        进而拖住整个服务的并发能力。
        """

        if (
            not self.settings.openai_api_key
            or self.settings.openai_api_key == "your-openai-api-key"
        ):
            raise ValueError(
                "OPENAI_API_KEY 未配置，请先在 .env 中设置后再调用 /api/chat"
            )

        request_url = (
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": messages,
            "temperature": 0.7,
        }

        # 为什么缓存键要基于 model + messages：
        # 因为“相同问题”在 AI 系统里并不总是意味着“相同答案”。
        # 如果上下文不同、模型不同，结果也可能不同。
        # 所以这里按最终完整请求做缓存，命中才是安全的。
        cache_payload = {
            "model": self.settings.openai_model,
            "messages": messages,
            "temperature": 0.7,
        }
        cached_answer = await self.cache_manager.get_json(
            namespace="llm_chat",
            payload=cache_payload,
        )
        if cached_answer is not None:
            return str(cached_answer), "hit"

        timeout = httpx.Timeout(timeout=float(self.settings.openai_timeout_seconds))

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    request_url,
                    headers=headers,
                    json=payload,
                )
        except httpx.RequestError as exc:
            raise ValueError(
                "调用 LLM 接口失败，请检查网络、OPENAI_BASE_URL 或服务可用性。"
            ) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ValueError(
                f"LLM 接口返回错误状态码：{response.status_code}，"
                "请检查 API Key、模型名或接口地址配置。"
            ) from exc

        response_json = response.json()

        try:
            answer = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "LLM 返回结果格式不符合预期，无法提取回答内容。"
            ) from exc

        if not answer:
            raise ValueError("LLM 已返回响应，但回答内容为空。")

        await self.cache_manager.set_json(
            namespace="llm_chat",
            payload=cache_payload,
            value=answer,
            ttl_seconds=self.settings.llm_cache_ttl_seconds,
        )

        return answer, "miss"
