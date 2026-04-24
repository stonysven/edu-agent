"""
这个文件的作用：
为 LangChain 版本的 RAG 提供一个独立的 Agent 封装。

为什么这里也单独加一个 Agent：
因为项目当前的结构是：
- pipeline / rag：负责底层能力
- agent：负责对外暴露更清晰的业务调用接口

既然原始 RAG 已经有 `RAGAgent`，
那么 LangChain 版本也保持同样的分层，会更整齐，也更方便路由层接入。
"""

from __future__ import annotations

from typing import Any

from app.rag.langchain_rag import LangChainRAG


class LangChainRAGAgent:
    """
    这个类的作用：
    包装 LangChainRAG，对外提供 Agent 风格的调用入口。
    """

    def __init__(self, rag_pipeline: LangChainRAG | None = None) -> None:
        self.rag_pipeline = rag_pipeline or LangChainRAG()
        self.agent_name = "langchain_rag_agent"

    async def load_knowledge_base(self, directory: str) -> dict[str, Any]:
        """
        这个方法的作用：
        构建 LangChain 版本的知识库向量索引。
        """

        return await self.rag_pipeline.build_vector_store(directory=directory)

    async def load_uploaded_files(self, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        """
        这个方法的作用：
        基于上传文件构建 LangChain 版本的知识库向量索引。
        """

        return await self.rag_pipeline.build_vector_store_from_uploaded_files(files=files)

    async def ask(self, question: str) -> dict[str, Any]:
        """
        这个方法的作用：
        执行一次 LangChain 版本的 RAG 问答。
        """

        result = await self.rag_pipeline.query(question=question)
        result["agent"] = self.agent_name
        return result
