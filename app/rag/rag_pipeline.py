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

概念解释注释块：为什么 token 优化在 RAG 中更重要
--------------------------------------------------
普通聊天里，token 主要来自系统提示词、历史和当前问题。
但在 RAG 里，还会额外叠加“检索出来的知识片段”。

这会导致一个常见问题：
如果检索片段过长、提示词又啰嗦，
那么 token 会增长得非常快。

它会直接带来两类代价：
1. 成本更高：
   检索片段越多、越长，每次问答就越贵。

2. 性能更差：
   prompt 越长，响应越慢，而且模型更容易被无关上下文干扰。

所以 RAG 优化的关键不是“塞更多上下文”，
而是“保留最相关、最紧凑的上下文”。
--------------------------------------------------
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.core.cache_manager import CacheManager, get_cache_manager
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
        cache_manager: CacheManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache_manager = cache_manager or get_cache_manager()
        self.document_loader = document_loader or DocumentLoader()
        self.text_splitter = text_splitter or TextSplitter(
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )
        self.embedding_client = embedding_client or EmbeddingClient(
            settings=self.settings,
            cache_manager=self.cache_manager,
        )
        self.vector_store = vector_store or InMemoryVectorStore(
            embedding_client=self.embedding_client,
        )
        if self.vector_store.embedding_client is None:
            self.vector_store.embedding_client = self.embedding_client

    async def load_knowledge_base(self, directory: str) -> dict[str, int | str]:
        """
        这个方法的作用：
        加载本地知识库目录，并完成：
        文档读取 -> 文本切分 -> embedding -> 向量存储
        """

        documents = self.document_loader.load_from_directory(directory=directory)
        return await self._index_documents(
            documents=documents,
            source_label=directory,
        )

    async def load_uploaded_files(self, files: list[tuple[str, bytes]]) -> dict[str, int | str]:
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
        return await self._index_documents(
            documents=documents,
            source_label=file_names or "uploaded_files",
        )

    async def _index_documents(
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

        # 为什么这里仍然保留受控并发：
        # 虽然 embedding 已经从“远程 API”改成了“本地模型”，
        # 但批量给很多 chunk 做向量化仍然会消耗 CPU/GPU 和内存。
        # 如果完全不限制并发，服务在导入大批文档时仍可能出现资源争抢。
        #
        # 所以这里继续保留一个可配置的并发上限，
        # 让教学项目在本地机器上也更稳定。
        semaphore = asyncio.Semaphore(self.settings.embedding_max_concurrency)

        async def build_stored_document(chunk: dict[str, str | int]) -> dict[str, object]:
            async with semaphore:
                embedding = await self.embedding_client.embed_text(str(chunk["text"]))
                return {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "embedding": embedding,
                }

        stored_documents = await asyncio.gather(
            *(build_stored_document(chunk) for chunk in chunks)
        )

        self.vector_store.add_documents(stored_documents)

        return {
            "directory": source_label,
            "document_count": len(documents),
            "chunk_count": len(stored_documents),
        }

    async def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """
        这个方法的作用：
        执行一次完整的 RAG 问答流程。
        """

        if self.vector_store.count() == 0:
            raise ValueError("知识库尚未加载，请先调用 /api/upload。")

        search_top_k = top_k or self.settings.rag_top_k
        retrieved_documents = await self.vector_store.similarity_search_by_text(
            query_text=question,
            top_k=search_top_k,
        )

        context = self._build_context(retrieved_documents)
        messages = self._build_rag_messages(context=context, question=question)
        prompt_length = self._estimate_messages_token_count(messages=messages)
        answer, cache_status = await self._call_llm_with_messages(messages=messages)

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
            {
                "step": "prompt_length",
                "content": str(prompt_length),
            },
            {
                "step": "history_ratio",
                "content": "0.00%",
            },
            {
                "step": "llm_cache",
                "content": cache_status,
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
                    f"[{index}] {item['source']}#chunk-{item['chunk_index']}\n"
                    f"{item['text']}"
                )
            )
        return "\n\n".join(context_blocks)

    def _build_rag_messages(
        self,
        context: str,
        question: str,
    ) -> list[dict[str, str]]:
        """
        这个方法的作用：
        为 RAG 问答构造更紧凑的 messages。

        为什么这里要单独抽出来：
        因为“提示词设计”和“接口调用”是两个不同职责。
        抽开后更容易单独优化 prompt，而不会影响网络请求代码。
        """

        prompt = f"基于以下知识回答。\n{context}\n\n问题：{question}"
        return [
            {
                "role": "system",
                "content": (
                    "你是知识库问答助手。"
                    "优先依据提供的知识回答；"
                    "证据不足时明确说明。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def _estimate_token_count(self, text: str) -> int:
        """
        这个方法的作用：
        对文本 token 数量做一个粗略估算。

        为什么这里继续用近似估算：
        因为当前目标是监控 token 趋势，而不是做严格计费。
        不引入额外 tokenizer，能保持实现简单。
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

    async def _call_llm_with_messages(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, str]:
        """
        这个方法的作用：
        把构造好的 RAG messages 交给 LLM，生成最终答案。
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

        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": messages,
            "temperature": 0.3,
        }

        cached_answer = await self.cache_manager.get_json(
            namespace="llm_rag",
            payload=payload,
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
                "调用 RAG 问答 LLM 接口失败，请检查网络、OPENAI_BASE_URL 或服务可用性。"
            ) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
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

        await self.cache_manager.set_json(
            namespace="llm_rag",
            payload=payload,
            value=answer,
            ttl_seconds=self.settings.llm_cache_ttl_seconds,
        )

        return answer, "miss"
