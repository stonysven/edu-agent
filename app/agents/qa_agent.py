"""
这个文件的作用：
实现 QAAgent，也就是“问答 Agent”。

它负责两种问答模式：
1. chat：普通聊天，直接调用 LLM
2. rag：基于知识库回答，调用 RAG pipeline

为什么把这两种能力都放进 QAAgent：
因为它们本质上都属于“回答用户问题”，
只是答案来源不同：
- chat 更依赖模型自身能力
- rag 更依赖知识库检索结果

这样做的好处是：
PlanningAgent 只需要判断“是不是问答，以及用哪种问答方式”，
而 QAAgent 只需要专注在“怎么把答案回答出来”。
"""

from __future__ import annotations

from app.agents.base_agent import AgentResult
from app.agents.rag_agent import RAGAgent
from app.agents.simple_agent import SimpleAgent


class QAAgent:
    """
    这个类的作用：
    作为统一的问答执行 Agent，对外暴露 chat 和 rag 两种模式。
    """

    def __init__(
        self,
        chat_agent: SimpleAgent | None = None,
        rag_agent: RAGAgent | None = None,
    ) -> None:
        self.chat_agent = chat_agent or SimpleAgent()
        self.rag_agent = rag_agent or RAGAgent()
        self.agent_name = "qa_agent"

    async def run_chat(self, user_message: str, session_id: str) -> AgentResult:
        """
        这个方法的作用：
        走普通聊天模式回答问题。
        """

        chat_result = await self.chat_agent.run(
            user_message=user_message,
            session_id=session_id,
        )
        return AgentResult(
            answer=chat_result.answer,
            agent=self.agent_name,
            session_id=session_id,
            intent="chat",
            trace=chat_result.trace,
            sources=[],
        )

    async def run_rag(self, user_message: str, session_id: str) -> AgentResult:
        """
        这个方法的作用：
        走 RAG 模式回答问题。
        """

        rag_result = await self.rag_agent.ask(question=user_message)
        return AgentResult(
            answer=rag_result["answer"],
            agent=self.agent_name,
            session_id=session_id,
            intent="rag",
            trace=rag_result["trace"],
            sources=rag_result["sources"],
        )
