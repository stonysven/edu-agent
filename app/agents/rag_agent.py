"""
这个文件的作用：
为 RAG 问答提供一个独立的 Agent 封装。

为什么要单独加一个 `rag_agent`：
因为从系统分层角度看，
“普通多轮对话 Agent”和“基于知识库问答的 Agent”职责并不完全相同。

把 RAG 问答单独封装成 Agent 后，
后续如果要做：
- 通用聊天 Agent
- 教学问答 Agent
- 文档检索 Agent
- 规划型 Agent

就会更自然。
"""

from __future__ import annotations

from typing import Any

from app.rag.rag_pipeline import RAGPipeline


class RAGAgent:
    """
    这个类的作用：
    包装 RAG pipeline，对外提供更清晰的 Agent 风格调用接口。
    """

    def __init__(self, rag_pipeline: RAGPipeline | None = None) -> None:
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.agent_name = "rag_agent"

    async def load_knowledge_base(self, directory: str) -> dict[str, Any]:
        """
        这个方法的作用：
        触发知识库加载流程。
        """

        return await self.rag_pipeline.load_knowledge_base(directory=directory)

    async def load_uploaded_files(self, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        """
        这个方法的作用：
        触发上传文件入库流程。
        """

        return await self.rag_pipeline.load_uploaded_files(files=files)

    async def ask(self, question: str) -> dict[str, Any]:
        """
        这个方法的作用：
        执行一次基于知识库的问答。
        """

        result = await self.rag_pipeline.ask(question=question)
        result["agent"] = self.agent_name
        return result
