"""
这个文件的作用：
把长文档切分成较小的文本块（chunk）。

为什么 RAG 不能直接把整篇文档拿去做 embedding：
1. 长文档语义太杂，做成一个向量后会变得不够精准
2. 检索时只能命中整篇，难以定位真正相关的局部内容
3. 发送给模型的上下文会过长，成本和噪声都更高

所以我们通常会先“切块”：
- 每块保留一个相对聚焦的主题
- 检索时命中更精确
- 拼接上下文时也更灵活

为什么 chunk 大小很重要：
chunk 太小：
- 容易丢失上下文
- 语义可能不完整

chunk 太大：
- 检索不够精细
- 无关内容太多

当前这个教学版先采用最简单的字符切分方式：
- 默认每 500 个字符一块
- 默认重叠 100 个字符

为什么加 overlap：
因为一个句子或一个知识点可能刚好跨越 chunk 边界。
有一点重叠后，能减少信息被硬切断的概率。
"""

from __future__ import annotations


Chunk = dict[str, str | int]


class TextSplitter:
    """
    这个类的作用：
    把文档切分成适合做 embedding 和检索的小块。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[dict[str, str]]) -> list[Chunk]:
        """
        这个方法的作用：
        把一批文档切分成多个 chunk，并保留来源信息。
        """

        chunks: list[Chunk] = []
        for document in documents:
            source = document["source"]
            text = document["text"]
            chunks.extend(self._split_single_document(source=source, text=text))
        return chunks

    def _split_single_document(self, source: str, text: str) -> list[Chunk]:
        """
        这个方法的作用：
        把单篇文档切成多个 chunk。
        """

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "source": source,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                    }
                )

            # 为什么不是直接 `start = end`：
            # 因为我们希望相邻 chunk 之间保留一小段重叠区域。
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        return chunks
