"""
这个文件的作用：
负责使用本地 embedding 模型，把文本转成可用于检索的向量。

什么是 embedding：
embedding 可以理解为“把文本压缩成一串能够表达语义的数字”。

为什么 RAG 需要 embedding：
RAG 的检索阶段并不是直接让计算机理解自然语言，
而是先把：
1. 知识库 chunk
2. 用户查询
都映射到同一个向量空间里，
然后再比较它们是否“语义接近”。

为什么这里改用本地模型：
你这次要求把 embedding 从远程 API 改成“本地可运行”的方案。
`sentence-transformers/all-MiniLM-L6-v2` 正是一个非常常见的本地语义向量模型。

这个模型在做什么：
1. 它会把一句文本送进 Transformer 编码器
2. 提取能表示语义的信息
3. 输出一个固定维度的向量

为什么它可以替代 OpenAI embedding：
因为 RAG 检索阶段真正需要的能力不是“必须由 OpenAI 生成向量”，
而是“能稳定地把语义相近的文本映射到相近的向量空间”。
`all-MiniLM-L6-v2` 正好具备这个能力，所以在教学项目、原型项目、
本地开发环境里，完全可以替代远程 OpenAI embedding 接口。

两者的主要差异是：
1. OpenAI embedding：
   - 依赖外部网络服务
   - 通常效果稳定，省去本地模型管理
   - 但会增加调用成本和网络依赖
2. sentence-transformers 本地 embedding：
   - 第一次运行会下载模型，之后本地即可复用
   - 无需每次请求外部服务
   - 很适合离线开发、教学和低成本原型

这也是为什么这里的替换是成立的：
只要“向量能表达语义相似性”，RAG 检索流程本身就可以继续工作。
"""

from __future__ import annotations

import asyncio

from sentence_transformers import SentenceTransformer

from app.core.cache_manager import CacheManager, get_cache_manager
from app.core.config import Settings, get_settings


class EmbeddingClient:
    """
    这个类的作用：
    封装本地 sentence-transformers embedding 模型。

    为什么仍然保留 `EmbeddingClient` 这个抽象名字：
    因为上层 RAG 流程并不需要关心“向量来自远程 API 还是本地模型”，
    只需要知道“这里有一个统一的 embedding 入口”即可。
    这样以后如果想再切换模型，实现成本会更低。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache_manager = cache_manager or get_cache_manager()

        # 为什么在初始化时加载模型：
        # 本地模型加载通常比单次 encode 更重。
        # 把模型实例缓存到对象里，后续每次 embedding 都能直接复用，
        # 避免重复加载模型文件造成明显延迟。
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    async def embed_text(self, text: str) -> list[float]:
        """
        这个方法的作用：
        为知识库里的普通文本生成 embedding 向量。

        为什么叫 `embed_text`：
        因为在 RAG 中，大量被索引的对象是“文档 chunk 文本”。
        这个命名能更直观地表达“把一段文本拿去做向量化”。
        """

        return await self._embed(text=text, cache_role="document")

    async def embed_query(self, text: str) -> list[float]:
        """
        这个方法的作用：
        为用户查询生成 embedding 向量。

        为什么要单独保留 `embed_query`：
        许多检索系统都会区分“文档向量”和“查询向量”的入口，
        因为未来如果你想换成专门的双塔检索模型，
        文档编码和查询编码可能会走不同 prompt 或不同头部。

        当前 `all-MiniLM-L6-v2` 对两者都可以直接使用同一个 encode，
        但接口先分开，后续扩展会更自然。
        """

        return await self._embed(text=text, cache_role="query")

    async def _embed(self, text: str, cache_role: str) -> list[float]:
        """
        这个方法的作用：
        统一处理本地模型 embedding、缓存和结果格式转换。
        """

        normalized_text = text.strip()
        if not normalized_text:
            return []

        payload = {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "role": cache_role,
            "text": normalized_text,
        }

        # 为什么本地模型也值得做缓存：
        # 虽然这里不再走远程 API 计费，但 embedding 计算本身仍然需要 CPU/GPU 时间。
        # 尤其是知识库重复导入、重复查询时，缓存命中能直接减少重复编码。
        cached_embedding = await self.cache_manager.get_json(
            namespace="embedding",
            payload=payload,
        )
        if cached_embedding is not None:
            return [float(item) for item in cached_embedding]

        # 为什么把 encode 放到线程里执行：
        # `SentenceTransformer.encode()` 是同步 CPU/GPU 计算。
        # 当前项目的 API 层是异步的，如果直接在协程里同步执行，
        # 会阻塞事件循环，影响并发请求的响应能力。
        embedding = await asyncio.to_thread(self._encode_text, normalized_text)

        await self.cache_manager.set_json(
            namespace="embedding",
            payload=payload,
            value=embedding,
            ttl_seconds=self.settings.embedding_cache_ttl_seconds,
        )

        return embedding

    def _encode_text(self, text: str) -> list[float]:
        """
        这个方法的作用：
        调用 sentence-transformers 模型，真正产出向量。

        模型在这里具体做的事情是：
        1. 对输入文本分词
        2. 通过预训练 Transformer 提取上下文语义特征
        3. 生成固定维度的句向量
        4. 返回 Python `list[float]`，便于后续序列化和缓存

        为什么这里能替代 OpenAI embedding：
        因为向量检索需要的是“语义可比较的向量表示”，
        而不是某个供应商专有的数据格式。
        只要文档和查询都由同一个 embedding 模型编码，
        相似度搜索就仍然成立。
        """

        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.astype(float).tolist()
