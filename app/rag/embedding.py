"""
这个文件的作用：
负责调用 embedding API，把文本转成向量。

什么是 embedding：
embedding 可以理解为“把文本映射成数字向量”。

为什么 RAG 需要 embedding：
因为计算机不能直接拿自然语言文本去做“语义距离”比较，
但可以比较两个向量在高维空间里的相似程度。

所以典型流程是：
1. 把知识库 chunk 转成向量
2. 把用户问题也转成向量
3. 比较两者的距离
4. 找出最相关的知识块

为什么 embedding 比关键词匹配更好：
因为关键词匹配只看字面重合，
而 embedding 更接近“语义相似”。
即使用户换了一种说法，也可能找到正确知识。
"""

from __future__ import annotations

from typing import Any

import requests

from app.core.config import Settings, get_settings


class EmbeddingClient:
    """
    这个类的作用：
    封装 embedding 接口调用逻辑。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def embed_text(self, text: str) -> list[float]:
        """
        这个方法的作用：
        把一段文本转成一个 embedding 向量。
        """

        # 为什么 embedding 不再直接复用聊天的 API Key：
        # 因为聊天模型和 embedding 模型可能来自不同供应商。
        # 当前你的使用场景就是：
        # - chat：智谱
        # - embedding：OpenAI
        #
        # 所以这里必须读取专门的 embedding 配置。
        if (
            not self.settings.embedding_api_key
            or self.settings.embedding_api_key == "your-openai-api-key"
        ):
            raise ValueError(
                "EMBEDDING_API_KEY 未配置，请先在 .env 中设置后再调用 embedding 接口"
            )

        request_url = f"{self.settings.embedding_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.embedding_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.settings.openai_embedding_model,
            "input": text,
        }

        try:
            response = requests.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=self.settings.embedding_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ValueError(
                "调用 embedding 接口失败，请检查网络、EMBEDDING_BASE_URL 或服务可用性。"
            ) from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ValueError(
                f"embedding 接口返回错误状态码：{response.status_code}，"
                "请检查 API Key、embedding 模型名或接口地址配置。"
            ) from exc

        response_json = response.json()

        try:
            embedding = response_json["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("embedding 接口返回结构不符合预期。") from exc

        return embedding
