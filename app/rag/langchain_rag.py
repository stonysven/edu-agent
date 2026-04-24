"""
这个文件的作用：
在保留当前“手写 RAG”实现的基础上，新增一个 LangChain 风格的 RAG 版本。

为什么要单独新增，而不是直接替换旧实现：
因为这个项目本身有很强的教学属性。
保留原始实现，能帮助我们看到：
1. 一个 RAG 系统的底层链路到底由哪些步骤组成
2. LangChain 究竟帮我们封装了哪些重复工程工作
3. 为什么工业项目常常更愿意使用成熟框架

这也是这次“渐进式改造”的目标：
- 原始 RAG：帮助理解原理
- LangChain RAG：帮助理解工业封装方式

LangChain 在这里做了什么封装：
1. `Document`：
   把“文本 + metadata”封装成统一对象，而不是我们手动维护字典结构
2. `HuggingFaceEmbeddings`：
   把本地 embedding 模型封装成一个统一接口，方便替换模型供应商
3. `FAISS`：
   帮我们封装向量索引、向量存储和相似度检索逻辑
4. Retriever 风格接口：
   让“检索 top_k 文档”变成统一调用，而不是手写相似度排序流程

它和手写 RAG 的核心区别：
1. 手写 RAG：
   - 每一步都显式可见
   - 非常适合学习“底层到底发生了什么”
   - 但工程代码会更分散，很多基础设施要自己维护
2. LangChain RAG：
   - 把通用步骤抽象成组件
   - 更容易组合和替换实现
   - 更接近工业实践
   - 但阅读时会多一层框架抽象

为什么这里仍然沿用项目现有的 LLM 调用方式：
因为当前目标是“渐进式引入 LangChain 优化 RAG”，
重点在于替换 embedding / vector store / retrieval 这些检索层封装。
如果连生成层也一起切成另一套框架，会让对比点变得不清楚。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.cache_manager import CacheManager, get_cache_manager
from app.core.config import Settings, get_settings
from app.rag.document_loader import DocumentLoader
from app.rag.text_splitter import TextSplitter


class LangChainRAG:
    """
    这个类的作用：
    提供一个基于 LangChain 组件封装的 RAG 实现。

    这个实现和项目原始 `RAGPipeline` 并行存在：
    - `RAGPipeline`：强调底层原理
    - `LangChainRAG`：强调工业封装

    为什么这种并行结构很有价值：
    因为团队可以先用原始实现理解系统，再用 LangChain 版本体验
    “少写多少底层基础设施代码”。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        document_loader: DocumentLoader | None = None,
        text_splitter: TextSplitter | None = None,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache_manager = cache_manager or get_cache_manager()
        self.document_loader = document_loader or DocumentLoader()
        self.text_splitter = text_splitter or TextSplitter(
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )

        # 为什么这里使用 LangChain 提供的 `HuggingFaceEmbeddings`：
        # 它把本地 sentence-transformers 模型封装成了 LangChain 统一的 embedding 接口。
        # 这样向量库组件只需要依赖“标准 embeddings 能力”，
        # 而不需要感知底层到底是 OpenAI、本地 HuggingFace，还是别的供应商。
        #
        # 当前我们继续使用本地 `all-MiniLM-L6-v2`，
        # 这样可以和项目现有本地 embedding 方案保持一致，方便公平对比。
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store: FAISS | None = None

    def load_documents(self, directory: str) -> list[Document]:
        """
        这个方法的作用：
        加载目录中的文档，并转换成 LangChain `Document` 列表。

        为什么这里仍然复用项目自己的 `DocumentLoader` 和 `TextSplitter`：
        因为这能确保：
        1. 输入文档来源一致
        2. 切分策略一致
        3. 对比“原始 RAG vs LangChain RAG”时更公平

        换句话说，这里变化的重点不是“换一套输入数据处理规则”，
        而是把 embedding / vector store / retrieval 这些中间层换成工业封装。
        """

        raw_documents = self.document_loader.load_from_directory(directory=directory)
        if not raw_documents:
            raise ValueError(f"没有可用文档可建立索引：{directory}")

        chunks = self.text_splitter.split_documents(raw_documents)
        if not chunks:
            raise ValueError(f"文档切分后没有得到可用 chunk：{directory}")

        langchain_documents: list[Document] = []
        for chunk in chunks:
            langchain_documents.append(
                Document(
                    page_content=str(chunk["text"]),
                    metadata={
                        "source": str(chunk["source"]),
                        "chunk_index": int(chunk["chunk_index"]),
                    },
                )
            )

        return langchain_documents

    def load_uploaded_documents(self, files: list[tuple[str, bytes]]) -> list[Document]:
        """
        这个方法的作用：
        把上传文件转换成 LangChain `Document` 列表。

        为什么这里也补齐上传文件入口：
        因为项目原始 RAG 已经支持“目录导入”和“上传导入”两种方式。
        如果 LangChain 版只支持目录，不支持上传，
        那么两套实现的对比就不够完整。
        """

        raw_documents = self.document_loader.load_from_uploaded_files(files=files)
        if not raw_documents:
            raise ValueError("没有可用上传文档可建立 LangChain 索引。")

        chunks = self.text_splitter.split_documents(raw_documents)
        if not chunks:
            raise ValueError("上传文档切分后没有得到可用 chunk。")

        langchain_documents: list[Document] = []
        for chunk in chunks:
            langchain_documents.append(
                Document(
                    page_content=str(chunk["text"]),
                    metadata={
                        "source": str(chunk["source"]),
                        "chunk_index": int(chunk["chunk_index"]),
                    },
                )
            )

        return langchain_documents

    async def build_vector_store(self, directory: str) -> dict[str, int | str]:
        """
        这个方法的作用：
        加载文档并构建 LangChain 版本的 FAISS 向量库。

        为什么这里选择 FAISS：
        因为它是本地向量索引里非常常见的一种方案，
        既适合作为教学项目里的“工业化升级版”，
        也足够轻量，能直接在本地运行。

        这里和手写版最大的区别是：
        - 手写版自己维护 `list[document] + cosine similarity`
        - LangChain 版把“向量存储 + 建索引 + 检索接口”交给 FAISS 组件统一处理
        """

        documents = self.load_documents(directory=directory)

        # 为什么用 `asyncio.to_thread`：
        # LangChain 构建 FAISS 索引的过程本质上是同步 CPU 计算。
        # 当前项目的 API 链路是异步的，所以把重计算放到线程里执行，
        # 可以避免阻塞事件循环。
        self.vector_store = await asyncio.to_thread(
            FAISS.from_documents,
            documents,
            self.embeddings,
        )

        return {
            "directory": directory,
            "document_count": len({document.metadata["source"] for document in documents}),
            "chunk_count": len(documents),
            "vector_store": "faiss",
        }

    async def build_vector_store_from_uploaded_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> dict[str, int | str]:
        """
        这个方法的作用：
        基于上传文件构建 LangChain 版本的 FAISS 向量库。
        """

        documents = self.load_uploaded_documents(files=files)
        self.vector_store = await asyncio.to_thread(
            FAISS.from_documents,
            documents,
            self.embeddings,
        )

        return {
            "directory": "uploaded_files",
            "document_count": len({document.metadata["source"] for document in documents}),
            "chunk_count": len(documents),
            "vector_store": "faiss",
        }

    async def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """
        这个方法的作用：
        使用 LangChain 向量库执行检索，然后沿用项目现有 LLM 调用方式生成答案。

        为什么这仍然算 LangChain RAG：
        因为 RAG 的“检索增强”核心主要发生在：
        1. 文档组织
        2. embedding
        3. 向量索引
        4. 相似度检索

        当前我们正是把这几层交给 LangChain 组件处理。
        生成层继续使用现有实现，是为了让对比更聚焦。
        """

        if self.vector_store is None:
            raise ValueError("LangChain 向量库尚未构建，请先调用 build_vector_store().")

        search_top_k = top_k or self.settings.rag_top_k
        retrieved_documents = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            question,
            search_top_k,
        )

        normalized_documents: list[dict[str, object]] = []
        for document, score in retrieved_documents:
            normalized_documents.append(
                {
                    "source": document.metadata.get("source", "unknown"),
                    "chunk_index": int(document.metadata.get("chunk_index", -1)),
                    "text": document.page_content,
                    # 为什么这里把分数统一叫做 `score`：
                    # 对外返回结构尽量和原始 RAG 保持一致，方便做 A/B 对比。
                    #
                    # 需要注意：
                    # FAISS 返回的 score 常常更接近“距离”而不是我们手写版里的“余弦相似度”。
                    # 所以两套实现的分数数值不适合直接做绝对比较，
                    # 更适合比较“排序结果”和最终回答质量。
                    "score": float(score),
                }
            )

        context = self._build_context(normalized_documents)
        messages = self._build_rag_messages(context=context, question=question)
        prompt_length = self._estimate_messages_token_count(messages=messages)
        answer, cache_status = await self._call_llm_with_messages(messages=messages)

        return {
            "answer": answer,
            "sources": normalized_documents,
            "trace": [
                {
                    "step": "rag_mode",
                    "content": "langchain_rag",
                },
                {
                    "step": "vector_store",
                    "content": "FAISS via LangChain",
                },
                {
                    "step": "retrieved_docs",
                    "content": str(
                        [
                            {
                                "source": item["source"],
                                "chunk_index": item["chunk_index"],
                            }
                            for item in normalized_documents
                        ]
                    ),
                },
                {
                    "step": "retrieval_scores",
                    "content": str(
                        [round(float(item["score"]), 6) for item in normalized_documents]
                    ),
                },
                {
                    "step": "prompt_length",
                    "content": str(prompt_length),
                },
                {
                    "step": "llm_cache",
                    "content": cache_status,
                },
            ],
        }

    def _build_context(self, retrieved_documents: list[dict[str, object]]) -> str:
        """
        这个方法的作用：
        把 LangChain 检索结果拼成可发送给 LLM 的上下文。

        为什么这里继续手动拼接上下文：
        因为这能让 LangChain 版和原始版使用尽量一致的提示词结构，
        从而把比较重点放在“检索层封装差异”上。
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
        构造 LangChain RAG 版本发送给 LLM 的消息。
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
        对文本 token 数量做粗略估算，便于和原始 RAG 版本对比。
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
        沿用项目当前已有的 LLM 调用方式生成最终答案。

        为什么这里不强行换成 LangChain 的 LLM 封装：
        因为这次目标是“让项目同时拥有底层实现和工业实现”，
        而不是一次性重写整条链路。
        只改检索层，更利于理解 LangChain 在 RAG 中到底帮了什么忙。
        """

        if (
            not self.settings.openai_api_key
            or self.settings.openai_api_key == "your-openai-api-key"
        ):
            raise ValueError(
                "OPENAI_API_KEY 未配置，请先在 .env 中设置后再调用 LangChain RAG 问答。"
            )

        request_url = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
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
            namespace="llm_langchain_rag",
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
                "调用 LangChain RAG 的 LLM 接口失败，请检查网络、OPENAI_BASE_URL 或服务可用性。"
            ) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ValueError(
                f"LangChain RAG 的 LLM 接口返回错误状态码：{response.status_code}，"
                "请检查 API Key、模型名或接口地址配置。"
            ) from exc

        response_json = response.json()

        try:
            answer = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("LangChain RAG 的 LLM 返回结构不符合预期。") from exc

        if not answer:
            raise ValueError("LangChain RAG 的 LLM 返回了空答案。")

        await self.cache_manager.set_json(
            namespace="llm_langchain_rag",
            payload=payload,
            value=answer,
            ttl_seconds=self.settings.llm_cache_ttl_seconds,
        )

        return answer, "miss"
