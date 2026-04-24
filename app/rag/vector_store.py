"""
这个文件的作用：
实现一个“内存版向量库”。

为什么现在不用外部向量数据库：
因为你明确要求“基础版，不依赖外部向量数据库”。
所以这一版先用 Python 内存结构把原理跑通。

虽然这不适合大规模生产，但它有两个非常重要的教学价值：
1. 能帮助我们真正理解“向量检索是怎么工作的”
2. 后续切换到 Milvus / pgvector / Elasticsearch 时，心里会更有底

概念解释注释块：为什么用 cosine similarity
--------------------------------------------------
cosine similarity（余弦相似度）是向量检索里最常见的相似度计算方式之一。

它的直觉可以理解为：
- 不是比较两个向量“有多长”
- 而是比较它们“方向有多接近”

为什么这很适合文本语义比较：
因为 embedding 的重点通常不是绝对大小，
而是“语义方向”是否相似。

余弦相似度越接近 1：
- 表示两个向量方向越相似
- 对应文本语义越接近
--------------------------------------------------
"""

from __future__ import annotations

import math

from app.rag.embedding import EmbeddingClient


StoredDocument = dict[str, object]


class InMemoryVectorStore:
    """
    这个类的作用：
    用 Python 列表在内存中保存文本 chunk 和它们的向量。
    """

    def __init__(self, embedding_client: EmbeddingClient | None = None) -> None:
        self._documents: list[StoredDocument] = []
        self.embedding_client = embedding_client

    def add_documents(self, documents: list[StoredDocument]) -> None:
        """
        这个方法的作用：
        把已经算好 embedding 的文档块存入内存向量库。
        """

        self._documents.extend(documents)

    def clear(self) -> None:
        """
        这个方法的作用：
        清空当前内存向量库。
        """

        self._documents.clear()

    async def similarity_search_by_text(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> list[StoredDocument]:
        """
        这个方法的作用：
        直接接收“原始查询文本”，并在内部使用本地 embedding 模型完成检索。

        为什么要给 `vector_store` 增加这个入口：
        因为这样向量库就不只是在“被动接收外部算好的 query embedding”，
        而是能明确体现“检索阶段使用的是本地 embedding 模型”。

        这里发生了两步：
        1. 用本地 `sentence-transformers/all-MiniLM-L6-v2` 把 query 编码成向量
        2. 再和文档向量计算余弦相似度

        这也正是“本地 embedding 替代 OpenAI embedding”后的完整检索链路。
        """

        if self.embedding_client is None:
            raise ValueError("vector_store 未配置 embedding_client，无法执行文本检索。")

        query_embedding = await self.embedding_client.embed_query(query_text)
        return self.similarity_search(query_embedding=query_embedding, top_k=top_k)

    def similarity_search(self, query_embedding: list[float], top_k: int = 3) -> list[StoredDocument]:
        """
        这个方法的作用：
        根据查询向量，找出最相似的若干文档块。
        """

        scored_documents: list[StoredDocument] = []
        for document in self._documents:
            document_embedding = document["embedding"]
            score = self._cosine_similarity(query_embedding, document_embedding)
            scored_documents.append(
                {
                    **document,
                    "score": score,
                }
            )

        scored_documents.sort(key=lambda item: item["score"], reverse=True)
        return scored_documents[:top_k]

    def count(self) -> int:
        """
        这个方法的作用：
        返回当前向量库中的 chunk 数量。
        """

        return len(self._documents)

    def _cosine_similarity(self, vector_a: list[float], vector_b: list[float]) -> float:
        """
        这个方法的作用：
        用纯 Python 标准库实现余弦相似度计算。

        为什么这里仍然可以继续复用原来的相似度逻辑：
        因为无论向量是来自 OpenAI embedding 还是本地 sentence-transformers，
        只要它们都是同一个向量空间中的数值向量，
        就可以继续使用余弦相似度比较语义接近程度。
        """

        if len(vector_a) != len(vector_b):
            raise ValueError("向量维度不一致，无法计算相似度。")

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
