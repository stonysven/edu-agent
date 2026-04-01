"""
这个文件的作用：
把 RAG 的完整流程串起来。

RAG 是什么：
RAG 是 Retrieval-Augmented Generation 的缩写，中文常叫“检索增强生成”。

它的核心思想不是“只问大模型”，而是：
1. 先从外部知识库里找相关内容
2. 再把这些内容交给大模型
3. 让模型基于这些知识来回答

为什么 RAG 比“直接问 LLM”更可靠：
1. 知识来源更可控：答案基于你提供的文档，而不是完全靠模型参数记忆
2. 可更新：你更新本地知识库后，不需要重新训练模型
3. 可追溯：可以返回引用来源，方便核对答案依据

这也是为什么 RAG 在企业知识库、课程问答、文档助手里非常常见。
"""

from __future__ import annotations

from typing import Any

import requests

from app.core.config import Settings, get_settings
from app.rag.document_loader import DocumentLoader
from app.rag.embedding import EmbeddingClient
from app.rag.text_splitter import TextSplitter
from app.rag.vector_store import InMemoryVectorStore


class RAGPipeline:
    """
    这个类的作用：
    封装完整的 RAG 流程。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        document_loader: DocumentLoader | None = None,
        text_splitter: TextSplitter | None = None,
        embedding_client: EmbeddingClient | None = None,
        vector_store: InMemoryVectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.document_loader = document_loader or DocumentLoader()
        self.text_splitter = text_splitter or TextSplitter(
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )
        self.embedding_client = embedding_client or EmbeddingClient(settings=self.settings)
        self.vector_store = vector_store or InMemoryVectorStore()

    def load_knowledge_base(self, directory: str) -> dict[str, int | str]:
        """
        这个方法的作用：
        加载本地知识库目录，并完成：
        文档读取 -> 文本切分 -> embedding -> 向量存储
        """

        documents = self.document_loader.load_from_directory(directory=directory)
        return self._index_documents(
            documents=documents,
            source_label=directory,
        )

    def load_uploaded_files(self, files: list[tuple[str, bytes]]) -> dict[str, int | str]:
        """
        这个方法的作用：
        接收上传文件内容，并完成：
        文件解析 -> 文本切分 -> embedding -> 向量存储

        为什么要提供这个方法：
        因为真正的产品和调试场景里，经常需要“临时上传一批文档马上提问”，
        而不是先手动把文件放到 `data/` 目录。
        """

        documents = self.document_loader.load_from_uploaded_files(files=files)
        file_names = ", ".join(document["source"] for document in documents)
        return self._index_documents(
            documents=documents,
            source_label=file_names or "uploaded_files",
        )

    def _index_documents(
        self,
        documents: list[dict[str, str]],
        source_label: str,
    ) -> dict[str, int | str]:
        """
        这个方法的作用：
        把“文档 -> chunk -> embedding -> 向量存储”这段公共流程收敛到一起。

        为什么要抽成私有方法：
        因为“从目录加载”和“从上传文件加载”只是输入来源不同，
        真正的索引流程是一样的。
        """

        if not documents:
            raise ValueError(f"没有可用文档可建立索引：{source_label}")

        chunks = self.text_splitter.split_documents(documents=documents)
        if not chunks:
            raise ValueError(f"文档切分后没有得到可用 chunk：{source_label}")

        self.vector_store.clear()

        stored_documents: list[dict[str, object]] = []
        for chunk in chunks:
            embedding = self.embedding_client.embed_text(str(chunk["text"]))
            stored_documents.append(
                {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "embedding": embedding,
                }
            )

        self.vector_store.add_documents(stored_documents)

        return {
            "directory": source_label,
            "document_count": len(documents),
            "chunk_count": len(stored_documents),
        }

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """
        这个方法的作用：
        执行一次完整的 RAG 问答流程。
        """

        if self.vector_store.count() == 0:
            raise ValueError("知识库尚未加载，请先调用 /api/upload。")

        search_top_k = top_k or self.settings.rag_top_k
        query_embedding = self.embedding_client.embed_text(question)
        retrieved_documents = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=search_top_k,
        )

        context = self._build_context(retrieved_documents)
        answer = self._call_llm_with_context(context=context, question=question)

        sources = [
            {
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "text": item["text"],
                "score": item["score"],
            }
            for item in retrieved_documents
        ]

        trace = [
            {
                "step": "retrieved_docs",
                "content": str(
                    [
                        {
                            "source": item["source"],
                            "chunk_index": item["chunk_index"],
                        }
                        for item in retrieved_documents
                    ]
                ),
            },
            {
                "step": "similarity_scores",
                "content": str(
                    [
                        round(float(item["score"]), 6)
                        for item in retrieved_documents
                    ]
                ),
            },
        ]

        return {
            "answer": answer,
            "sources": sources,
            "trace": trace,
        }

    def _build_context(self, retrieved_documents: list[dict[str, object]]) -> str:
        """
        这个方法的作用：
        把检索到的 chunk 拼成模型可读的上下文。
        """

        context_blocks: list[str] = []
        for index, item in enumerate(retrieved_documents, start=1):
            context_blocks.append(
                (
                    f"[片段 {index}]\n"
                    f"来源：{item['source']}#chunk-{item['chunk_index']}\n"
                    f"内容：{item['text']}"
                )
            )
        return "\n\n".join(context_blocks)

    def _call_llm_with_context(self, context: str, question: str) -> str:
        """
        这个方法的作用：
        把检索到的知识上下文和用户问题一起交给 LLM，生成最终答案。
        """

        if (
            not self.settings.openai_api_key
            or self.settings.openai_api_key == "your-openai-api-key"
        ):
            raise ValueError(
                "OPENAI_API_KEY 未配置，请先在 .env 中设置后再调用 /api/ask"
            )

        request_url = (
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        )
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        prompt = f"基于以下知识回答：\n{context}\n\n问题：{question}"

        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个基于知识库回答问题的教学型 AI 助手。"
                        "请尽量依据提供的知识片段回答，不要脱离上下文随意发挥。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.3,
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
                "调用 RAG 问答 LLM 接口失败，请检查网络、OPENAI_BASE_URL 或服务可用性。"
            ) from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ValueError(
                f"RAG 问答接口返回错误状态码：{response.status_code}，"
                "请检查 API Key、模型名或接口地址配置。"
            ) from exc

        response_json = response.json()

        try:
            answer = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("RAG 问答接口返回结构不符合预期。") from exc

        if not answer:
            raise ValueError("RAG 问答接口返回了空答案。")

        return answer
