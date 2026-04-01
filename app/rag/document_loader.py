"""
这个文件的作用：
从本地知识库目录中加载文档内容，作为 RAG 系统的输入。

为什么 RAG 需要先“加载文档”：
因为 RAG 的核心思想不是只靠大模型自己记忆知识，
而是先把外部知识读进系统，再在回答问题时进行检索。

如果连文档都没有先加载进来，
后面的切分、embedding、相似度检索都无从谈起。

当前这个教学版实现支持：
- `.txt`
- `.md`

为什么先只支持这两种：
因为它们最简单，足够适合作为第一版知识库输入。
等后续系统稳定后，再扩展 PDF、Word、网页抓取等格式会更合适。
"""

from __future__ import annotations

from pathlib import Path


Document = dict[str, str]


class DocumentLoader:
    """
    这个类的作用：
    负责从指定目录读取本地文档。
    """

    SUPPORTED_SUFFIXES = {".txt", ".md"}

    def load_from_directory(self, directory: str) -> list[Document]:
        """
        这个方法的作用：
        从本地目录中加载所有支持的文档文件。

        返回的数据结构中会保留：
        - `source`：文档来源路径
        - `text`：文档正文
        """

        directory_path = Path(directory)
        if not directory_path.exists():
            raise ValueError(f"知识库目录不存在：{directory}")

        if not directory_path.is_dir():
            raise ValueError(f"给定路径不是目录：{directory}")

        documents: list[Document] = []
        for file_path in sorted(directory_path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                continue

            text = file_path.read_text(encoding="utf-8")
            if not text.strip():
                # 空文档对检索没有价值，所以这里直接跳过。
                continue

            documents.append(
                {
                    "source": str(file_path),
                    "text": text,
                }
            )

        return documents

    def load_from_uploaded_files(self, files: list[tuple[str, bytes]]) -> list[Document]:
        """
        这个方法的作用：
        把上传的文件列表转换成统一的文档结构。

        参数说明：
        - `files`：每一项是 `(文件名, 文件字节内容)`

        为什么这里仍然复用和目录加载相同的返回结构：
        因为后面的切分、embedding、向量存储并不关心文档来自磁盘还是上传，
        它们只关心有没有统一的 `source + text` 结构。
        """

        documents: list[Document] = []
        for filename, file_bytes in files:
            suffix = Path(filename).suffix.lower()
            if suffix not in self.SUPPORTED_SUFFIXES:
                raise ValueError(f"不支持的文件类型：{filename}")

            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"文件不是有效的 UTF-8 文本：{filename}") from exc

            if not text.strip():
                continue

            documents.append(
                {
                    "source": filename,
                    "text": text,
                }
            )

        return documents
