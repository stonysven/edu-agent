"""
这个文件的作用：
定义当前项目的多 Agent 编排器（Orchestrator）。

为什么多 Agent 系统里必须有 Orchestrator：
因为当系统里不止一个 Agent 时，
就需要一个统一的“调度中心”来决定：
- 先做什么
- 该调用哪个 Agent
- 最后怎么把结果统一返回

概念解释注释块：Orchestrator 的作用
--------------------------------------------------
Orchestrator 可以理解为“总调度器”。

它本身不一定亲自回答问题，
但它负责把整个流程串起来：
1. 收到用户输入
2. 调用 PlanningAgent 判断意图
3. 选择 QAAgent 或 ToolAgent
4. 把结果整理成统一结构返回

为什么这样做比“让一个 Agent 全做完”更好：
因为每个 Agent 都可以专注在自己的职责上：
- PlanningAgent 负责判断
- QAAgent 负责回答
- ToolAgent 负责执行工具

这会让系统更清晰，也更容易扩展。
--------------------------------------------------

概念解释注释块：如何扩展更多 Agent
--------------------------------------------------
当前我们实现的是：
- PlanningAgent
- QAAgent
- ToolAgent

以后如果要加新的能力，比如：
- SearchAgent
- CodeAgent
- MemoryAgent

通常只需要两步：
1. 增加新的 Agent 文件
2. 在 PlanningAgent 和 Orchestrator 中加一条新的路由规则

这就是多 Agent 架构的扩展价值所在。
--------------------------------------------------
"""

from __future__ import annotations

from uuid import uuid4

from app.agents.base_agent import AgentResult
from app.agents.planning_agent import PlanningAgent
from app.agents.qa_agent import QAAgent
from app.agents.tool_agent import ToolAgent


class Orchestrator:
    """
    这个类的作用：
    作为当前系统的多 Agent 总编排入口。
    """

    def __init__(self) -> None:
        self.planning_agent = PlanningAgent()
        self.qa_agent = QAAgent()
        self.tool_agent = ToolAgent()

    async def handle_chat(self, user_message: str, session_id: str | None = None) -> AgentResult:
        """
        这个方法的作用：
        接收用户输入，先做规划，再分发到合适的 Agent。
        """

        current_session_id = session_id or str(uuid4())

        # 为什么这里需要知道 RAG 是否可用：
        # 因为如果知识库根本还没加载，就不应该把请求分给 RAG 路径。
        rag_available = self.qa_agent.rag_agent.rag_pipeline.vector_store.count() > 0

        planning_result = await self.planning_agent.plan(
            user_message=user_message,
            rag_available=rag_available,
        )

        intent = planning_result["intent"]
        reason = planning_result["reason"]

        planning_trace = [
            {
                "step": "planning_step",
                "content": f"intent={intent}，reason={reason}",
            }
        ]

        if intent == "tool":
            agent_trace = {
                "step": "agent_selection",
                "content": "根据 PlanningAgent 的判断，选择 ToolAgent 执行工具请求。",
            }
            result = await self.tool_agent.run(
                user_message=user_message,
                session_id=current_session_id,
            )
        elif intent == "rag":
            agent_trace = {
                "step": "agent_selection",
                "content": "根据 PlanningAgent 的判断，选择 QAAgent 的 RAG 模式回答问题。",
            }
            result = await self.qa_agent.run_rag(
                user_message=user_message,
                session_id=current_session_id,
            )
        else:
            agent_trace = {
                "step": "agent_selection",
                "content": "根据 PlanningAgent 的判断，选择 QAAgent 的普通聊天模式。",
            }
            result = await self.qa_agent.run_chat(
                user_message=user_message,
                session_id=current_session_id,
            )

        result.trace = planning_trace + [agent_trace] + result.trace
        return result
