"""
这个文件的作用：
定义对外提供的 HTTP 路由。

为什么把路由单独放在 `api` 层：
因为 API 层应该只做三件事：
1. 接收请求
2. 校验输入
3. 返回标准输出

真正的智能体逻辑、RAG 逻辑、记忆逻辑，不应该直接堆在这里，
否则接口文件会越来越臃肿，后续难以维护。

概念解释注释块：Agent / Trace 是什么
--------------------------------------------------
在 AI Agent 系统里，一个 `/chat` 接口通常不只是“把问题丢给模型”。
它背后可能会经历这些步骤：
1. 先判断用户意图
2. 再决定是否检索知识库（RAG）
3. 再决定是否读取历史记忆（Memory）
4. 再决定是否调用外部工具（Tools）
5. 最后由某个 Agent 组织答案返回

其中：
- `agent` 字段：表示本次请求是由哪个智能体负责处理
- `trace` 字段：表示本次执行过程中的关键轨迹

为什么现在先返回一个简单的固定结构：
因为项目骨架阶段，目标是先把“接口契约”稳定下来。
等后续接入真正的 Agent 编排逻辑时，前端和调用方不需要大改。
--------------------------------------------------
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.orchestrator.orchestrator import Orchestrator

# 为什么先创建一个独立的 APIRouter：
# 因为随着接口增多，我们不希望所有路由都堆在 `main.py` 中。
# 使用路由对象可以让不同业务模块各自维护自己的接口。
router = APIRouter(tags=["chat"])

# 为什么在模块级别创建一个编排器实例：
# 因为当前项目还很小，这样写最简单、最直观。
# 后续如果系统变复杂，再演进为依赖注入或应用生命周期管理即可。
orchestrator = Orchestrator()


class ChatRequest(BaseModel):
    """
    这个类的作用：
    定义 `/chat` 接口的请求体结构。

    为什么这里先只保留一个 `message` 字段：
    对教学项目来说，先保持最小输入会更容易理解。
    后续如果要支持：
    - session_id
    - user_id
    - metadata
    - conversation_history
    都可以在这个类里继续扩展。
    """

    # `...` 的含义是“这是必填字段”。
    # 也就是说，请求里如果没有 `message`，FastAPI 会自动返回参数校验错误。
    message: str = Field(..., description="用户输入的消息内容")


class ChatResponse(BaseModel):
    """
    这个类的作用：
    定义 `/chat` 接口的标准返回结构。

    为什么要显式定义响应模型：
    1. 接口契约更明确
    2. Swagger 文档更清楚
    3. 调用方能稳定依赖返回结构

    字段说明：
    - answer：智能体最终给用户的回答
    - agent：处理本次请求的智能体名称
    - trace：执行过程轨迹，后续可用于调试、教学或可观测性
    """

    answer: str
    agent: str
    trace: list[Any]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    这个函数的作用：
    处理用户的聊天请求，并返回标准 JSON 结果。

    为什么这里先用最简单的实现：
    当前阶段我们更关心“接口先跑起来、结构先稳定”，
    而不是一上来就接入复杂的 Agent 调度器、记忆系统和 RAG 流程。

    如果现在就引入复杂实现，会有什么问题：
    - 新手不容易看懂
    - 调试路径会变长
    - 系统边界还没稳定时，过度抽象容易造成返工

    这里做了什么：
    1. 接收已经通过 Pydantic 校验的请求对象
    2. 把请求交给 Orchestrator
    3. 由 Orchestrator 调用具体 Agent
    4. 返回统一的 `answer / agent / trace` 结构
    """

    # 为什么现在改为调用编排器：
    # 因为 /chat 接口本身不应该直接承担智能体逻辑。
    # 接口层只负责接收输入，然后把任务交给系统内部的编排层。
    try:
        result = orchestrator.handle_chat(user_message=request.message)
    except ValueError as exc:
        # 为什么把已知错误转换成 HTTPException：
        # 因为 API 层的职责之一，就是把内部错误转换成调用方更容易理解的 HTTP 响应。
        # 对当前项目来说，ValueError 通常意味着配置问题、请求问题或上游模型调用问题。
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 最后把编排器返回的统一结果封装成 API 响应模型并返回。
    return ChatResponse(
        answer=result.answer,
        agent=result.agent,
        trace=result.trace,
    )
