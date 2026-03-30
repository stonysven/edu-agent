"""
这个文件的作用：
实现项目中的第一个“可用 Agent”。

为什么叫 `simple_agent`：
因为它是一个最小版本的 Agent，
目标不是做复杂规划，而是先打通完整链路：

1. 接收用户输入
2. 调用 LLM
3. 生成真实答案
4. 返回 trace

为什么这个文件很重要：
它是从“项目骨架”走向“真正可用 AI Agent”的第一步。
虽然现在它只调用一次 LLM，
但它已经具备了 Agent 的基本形态。

概念解释注释块：trace 的意义
--------------------------------------------------
trace 可以理解为“Agent 执行过程的可观察记录”。

为什么 AI Agent 系统需要 trace：
1. 方便调试：知道模型前后做了什么
2. 方便教学：让初学者看到 Agent 的工作流程
3. 方便扩展：以后接入工具调用、记忆、RAG 时，trace 会更有价值

在当前最小版本里，我们人为构造 3 个关键步骤：
- thought：Agent 对用户问题的初步理解
- llm_call：Agent 准备调用大模型
- final：Agent 拿到最终答案并返回

注意：
这里的 trace 不是模型内部真实的“隐式推理链”。
它是系统层面的“显式执行轨迹”。
这样既更安全，也更适合作为工程日志与教学输出。
--------------------------------------------------

概念解释注释块：LLM 调用流程
--------------------------------------------------
一次最基础的 LLM 调用，通常会经历这些步骤：

1. 从配置中读取 API Key、模型名、接口地址
2. 构造请求头
3. 构造请求体
4. 通过 HTTP POST 请求发送给模型服务
5. 解析响应 JSON
6. 从响应中提取最终文本答案

为什么这里使用 `requests`：
因为它足够简单、直观，很适合作为教学级起点。
后续如果项目变复杂，再考虑引入更高级的客户端封装也不迟。
--------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import requests

from app.agents.base_agent import AgentResult, BaseAgent
from app.core.config import Settings, get_settings


class SimpleAgent(BaseAgent):
    """
    这个类的作用：
    实现最简单的单 Agent 对话能力。

    它当前的职责非常清晰：
    1. 理解用户输入的大致意图
    2. 调用 LLM 生成回答
    3. 构造 trace 并返回统一结果

    为什么把 LLM 调用封装到 Agent 内部：
    因为在当前最小版本里，Agent 自己就承担“思考 + 调用模型 + 组织输出”的职责。
    等后续系统规模变大，再把 LLM Client 抽到独立模块会更合适。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        这个初始化方法的作用：
        为 Agent 准备运行时所需的配置。

        为什么允许外部传入 `settings`：
        这样后续做测试或替换配置来源时会更灵活。

        为什么没有强制用户传入：
        因为默认从全局配置读取，是当前最简单、最自然的使用方式。
        """

        self.settings = settings or get_settings()
        self.agent_name = "simple_agent"

    def run(self, user_message: str) -> AgentResult:
        """
        这个方法的作用：
        执行一次最小 Agent 对话流程，并返回标准结果。

        这里的执行流程分为 3 步：
        1. 先构造 thought，表示 Agent 对输入的初步理解
        2. 再调用 LLM 生成真实回答
        3. 最后补充 final trace 并返回结果

        为什么流程保持这么简单：
        因为这是“第一版能跑通的 Agent”。
        当前重点是把主链路打通，而不是一开始就做复杂规划器。
        """

        trace: list[dict[str, str]] = []

        # 为什么先写一条 thought：
        # 因为 Agent 系统通常不会直接“无痕”返回结果，
        # 我们希望调用方能看到系统是如何理解这个问题的。
        thought_content = self._build_thought(user_message=user_message)
        trace.append({"step": "thought", "content": thought_content})

        # 在真正发起大模型调用之前，先记录一条 llm_call。
        # 这样后续如果需要排查“问题出在模型前还是模型后”，会更清楚。
        trace.append(
            {
                "step": "llm_call",
                "content": f"调用 LLM 生成回答，模型为 {self.settings.openai_model}",
            }
        )

        # 这里真正执行 HTTP 请求，向 OpenAI 或兼容接口发起调用。
        answer = self._call_llm(user_message=user_message)

        # 拿到最终答案后，再写入 final 步骤。
        # 这样整个 trace 就形成了一个完整闭环。
        trace.append({"step": "final", "content": answer})

        return AgentResult(
            answer=answer,
            agent=self.agent_name,
            trace=trace,
        )

    def _build_thought(self, user_message: str) -> str:
        """
        这个方法的作用：
        根据用户输入构造一个简单的“思考描述”。

        为什么这里只做简单描述，而不做复杂意图识别：
        因为当前版本的目标是教学和打基础。
        如果现在就加入复杂分类器或多阶段分析，理解成本会明显上升。
        """

        return f"用户正在咨询：{user_message}"

    def _call_llm(self, user_message: str) -> str:
        """
        这个方法的作用：
        通过 HTTP 请求调用 LLM，并提取最终文本答案。

        为什么把这段逻辑单独抽成函数：
        因为“调用 LLM”是一个清晰、独立的职责。
        抽出来后，`run()` 方法会更清楚，
        未来如果切换模型提供商，也更容易修改。

        调用流程说明：
        1. 先检查 API Key 是否存在
        2. 构造请求 URL
        3. 构造请求头 headers
        4. 构造请求体 payload
        5. 发送 POST 请求
        6. 检查 HTTP 状态码
        7. 解析 JSON
        8. 提取 `choices[0].message.content`

        为什么这里使用 Chat Completions 兼容格式：
        - 结构简单
        - 很多 OpenAI 兼容服务也支持
        - 作为教学项目更容易理解

        这里是一个工程判断：
        虽然 OpenAI 官方推荐新项目逐步使用 Responses API，
        但当前这个最小版本更强调“兼容、简单、容易看懂”，
        所以先选择更直观的 chat/completions 格式。
        后续如果需要，我们可以再平滑切换。
        """

        # 为什么先检查 Key：
        # 如果没有 API Key，继续发请求只会得到更难理解的远端错误。
        # 提前在本地报错，能让问题更直接。
        if (
            not self.settings.openai_api_key
            or self.settings.openai_api_key == "your-openai-api-key"
        ):
            raise ValueError(
                "OPENAI_API_KEY 未配置，请先在 .env 中设置后再调用 /api/chat"
            )

        # 为什么把 URL 单独拼出来：
        # 因为后续如果要切换到兼容服务地址，
        # 只需要修改配置，不需要改业务代码。
        request_url = (
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        )

        # 为什么请求头要包含 Authorization：
        # OpenAI 与大多数兼容接口都通过 Bearer Token 识别调用身份。
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        # 为什么要使用 system + user 两类消息：
        # 这是 chat/completions 的常见组织方式。
        # - system：定义助手应该扮演什么角色
        # - user：传入用户真实问题
        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 edu-agent 项目中的第一个教学型 AI Agent。"
                        "你的回答要清晰、准确、友好，尽量帮助用户理解问题。"
                    ),
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            # 为什么加 temperature：
            # 这是一个常见的生成参数。
            # 这里选择相对稳妥的 0.7，既保留一定表达灵活性，也不过于发散。
            "temperature": 0.7,
        }

        # 为什么显式设置 timeout：
        # 网络请求如果没有超时限制，最坏情况下可能一直卡住。
        # 在服务端代码里，超时几乎总是必要的基础保护。
        try:
            response = requests.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=self.settings.openai_timeout_seconds,
            )
        except requests.RequestException as exc:
            # 为什么要单独捕获网络异常：
            # 因为网络错误和“模型返回了业务错误”不是一回事。
            # 单独处理后，调用方更容易区分是连不上服务，还是请求内容有问题。
            raise ValueError(
                "调用 LLM 接口失败，请检查网络、OPENAI_BASE_URL 或服务可用性。"
            ) from exc

        # `raise_for_status()` 的作用是：
        # 如果 HTTP 状态码不是 2xx，就直接抛出异常。
        # 这样可以更早暴露问题，而不是在后面的 JSON 解析阶段才发现异常。
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ValueError(
                f"LLM 接口返回错误状态码：{response.status_code}，"
                "请检查 API Key、模型名或接口地址配置。"
            ) from exc

        # 把响应体解析成 Python 字典，方便后续提取字段。
        response_json = response.json()

        # 为什么要做防御式解析：
        # 因为不同兼容服务虽然大体格式相似，
        # 但细节上可能存在差异。
        # 这里先按标准结构提取，如果结构异常，就抛出更明确的错误。
        try:
            answer = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "LLM 返回结果格式不符合预期，无法提取回答内容。"
            ) from exc

        # 为什么还要判断 answer 是否为空：
        # 有些服务虽然返回了结构，但内容可能为空。
        # 提前检查可以避免把空结果继续传给上层。
        if not answer:
            raise ValueError("LLM 已返回响应，但回答内容为空。")

        return answer
