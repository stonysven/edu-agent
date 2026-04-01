"""
这个文件的作用：
实现项目中的第一个“可用 Agent”，并让它支持多轮对话。

为什么叫 `simple_agent`：
因为它依然是一个最小版本的 Agent，
但现在比第一版多了一个重要能力：
- 可以读取并使用历史对话

这意味着它已经具备了最基础的“上下文连续性”。

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

为什么这里使用 `requests`：
因为它足够简单、直观，很适合作为教学级起点。
--------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import requests

from app.agents.base_agent import AgentResult, BaseAgent
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
        self.agent_name = "simple_agent"

    def run(self, user_message: str, session_id: str) -> AgentResult:
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

        thought_content = self._build_thought(
            user_message=user_message,
            history_count=len(recent_history),
        )
        trace.append({"step": "thought", "content": thought_content})

        trace.append(
            {
                "step": "history_used",
                "content": str(len(recent_history)),
            }
        )

        trace.append(
            {
                "step": "llm_call",
                "content": f"调用 LLM 生成回答，模型为 {self.settings.openai_model}",
            }
        )

        messages = self._build_messages(
            recent_history=recent_history,
            user_message=user_message,
        )

        prompt_length = self._estimate_prompt_length(messages=messages)
        trace.append(
            {
                "step": "prompt_length",
                "content": str(prompt_length),
            }
        )

        answer = self._call_llm(messages=messages)

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

    def _build_messages(
        self,
        recent_history: list[dict[str, str]],
        user_message: str,
    ) -> list[dict[str, str]]:
        """
        这个方法的作用：
        把“系统提示词 + 历史消息 + 当前用户消息”拼成最终发给 LLM 的消息列表。
        """

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "你是 edu-agent 项目中的第一个教学型 AI Agent。"
                    "你的回答要清晰、准确、友好，尽量帮助用户理解问题。"
                    "在回答时，请结合对话上下文，保持多轮对话连贯。"
                ),
            }
        ]

        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _estimate_prompt_length(self, messages: list[dict[str, str]]) -> int:
        """
        这个方法的作用：
        对 prompt 长度做一个简单估算。

        为什么这里只做字符级估算：
        因为真正精确的 token 计算需要专门分词器，
        这会引入额外依赖和复杂度。
        当前目标是让开发者“感知上下文变长了多少”，
        所以简单估算已经够用了。
        """

        total_length = 0
        for message in messages:
            total_length += len(message.get("role", ""))
            total_length += len(message.get("content", ""))
        return total_length

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """
        这个方法的作用：
        通过 HTTP 请求调用 LLM，并提取最终文本答案。
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

        try:
            response = requests.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=self.settings.openai_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ValueError(
                "调用 LLM 接口失败，请检查网络、OPENAI_BASE_URL 或服务可用性。"
            ) from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
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

        return answer
