"""
这个文件的作用：
统一管理项目配置，避免在各个文件里到处直接读取环境变量。

为什么要这样做：
1. 配置集中管理后，项目结构更清楚
2. 参数来源一致，更容易排查问题
3. 后续新增数据库、向量库、模型配置时，也能放在同一个地方

为什么这里选择 `pydantic-settings`：
因为它和 FastAPI / Pydantic 生态天然契合，写法直观，适合教学项目。
相比手动 `os.getenv()` 到处读取，这种方式更规范，也更容易扩展。
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 为什么在模块加载时先读取 `.env`：
# 因为很多本地开发环境不会提前把环境变量注入系统，
# 所以我们先把 `.env` 文件中的内容加载进当前进程。
# 这样后续读取 `OPENAI_API_KEY` 时就能拿到值。
#
# 为什么这里要加 `override=True`：
# 我们在实际排查中发现，有些终端环境里可能已经存在一个“空的”
# `OPENAI_API_KEY` 环境变量。
#
# `python-dotenv` 默认不会覆盖已经存在的环境变量，
# 即使那个已有变量是空字符串也不会覆盖。
# 这会导致：
# 1. `.env` 文件里明明写了真实 Key
# 2. 但应用读到的仍然是终端里那个空值
#
# 对当前这个教学项目来说，更符合直觉的行为是：
# “项目根目录中的 `.env` 应该优先作为本项目的运行配置”。
# 因此这里显式开启覆盖，避免空壳环境变量把真实配置挡住。
load_dotenv(override=True)


class Settings(BaseSettings):
    """
    这个类的作用：
    用统一、类型安全的方式管理应用配置。

    为什么用类来表示配置：
    因为类可以把相关配置聚合在一起，
    并且通过类型标注告诉开发者“这个配置应该是什么类型”。

    在 AI Agent 系统中的角色：
    配置类相当于系统的“基础开关面板”。
    后续无论是大模型 Key、向量库地址、记忆模块开关，还是工具超时设置，
    都可以从这里统一读取。
    """

    # 为什么保留应用名：
    # 应用名常用于文档标题、日志输出、监控标识。
    # 这里给一个默认值，能让项目在最小配置下直接启动。
    app_name: str = "edu-agent"

    # 为什么通过 alias 绑定环境变量：
    # 因为 Python 代码通常使用小写下划线风格，
    # 而环境变量通常使用全大写风格。
    # alias 可以让两种风格和平共处，代码可读性更好。
    app_env: str = Field(default="development", alias="APP_ENV")

    # 为什么默认给空字符串而不是直接报错：
    # 项目骨架阶段，我们希望新手先能跑起来，再逐步接入真实模型服务。
    # 如果一开始就强制要求 Key，学习门槛会更高。
    # 后续正式接入大模型时，可以再改成更严格的校验。
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # 为什么增加接口地址配置：
    # 因为你要求“使用 OpenAI API 或兼容接口”。
    # 把地址做成配置后，既能默认使用 OpenAI，
    # 也能很方便切换到兼容服务。
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
    )

    # 为什么增加模型名配置：
    # 因为后续不同环境可能会使用不同模型。
    # 做成配置能避免把模型名写死在业务代码里。
    openai_model: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_MODEL",
    )

    # 为什么增加超时配置：
    # 网络调用总会有失败或延迟风险。
    # 把超时提成配置后，后续可以按环境灵活调整。
    openai_timeout_seconds: int = Field(
        default=60,
        alias="OPENAI_TIMEOUT_SECONDS",
    )

    # 为什么增加 Redis 配置：
    # 因为这一版要支持把对话历史优先存到 Redis 中。
    # 这样服务重启后历史不容易丢失，也更适合未来扩展成多实例。
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
    )

    # 为什么把历史轮数也做成配置：
    # 因为上下文长度策略是一个很常见、也很重要的可调参数。
    # 做成配置后，后续调优不需要改代码。
    memory_history_limit: int = Field(
        default=5,
        alias="MEMORY_HISTORY_LIMIT",
    )

    # 为什么增加 embedding 模型配置：
    # 因为聊天模型和 embedding 模型通常不是同一个。
    # RAG 检索依赖 embedding，所以必须把它单独配置出来。
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )

    # 为什么还要把 embedding 的 API Key 和 Base URL 单独拆出来：
    # 因为真实项目里，聊天模型和 embedding 模型不一定来自同一家服务。
    # 例如：
    # - chat 走智谱兼容接口
    # - embedding 走 OpenAI 官方接口
    #
    # 如果它们共用同一组配置，会导致：
    # - chat 可用，但 embedding 调错服务
    # - 或 embedding 可用，但 chat 调错服务
    #
    # 所以这里把 embedding 的认证信息和地址独立出来，
    # 让两条链路可以完全分开配置。
    embedding_api_key: str = Field(
        default="",
        alias="EMBEDDING_API_KEY",
    )
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="EMBEDDING_BASE_URL",
    )

    # 为什么保留一个 embedding timeout：
    # 虽然当前可以和聊天共用超时，但单独拆出来更利于后续微调。
    embedding_timeout_seconds: int = Field(
        default=60,
        alias="EMBEDDING_TIMEOUT_SECONDS",
    )

    # 为什么增加并发与限流相关配置：
    # AI 服务的单次请求通常比普通 CRUD 请求更重。
    # 一个请求里可能包含：
    # - 外部 LLM 调用
    # - embedding 调用
    # - 检索
    # - 多 Agent 编排
    #
    # 如果完全不做并发保护，高峰期很容易把：
    # 1. 本地 worker
    # 2. 上游模型接口
    # 3. 网络连接池
    # 一起拖慢。
    #
    # 所以这里把“最大并发请求数”和“时间窗口限流”都做成配置。
    max_concurrent_requests: int = Field(
        default=20,
        alias="MAX_CONCURRENT_REQUESTS",
    )
    rate_limit_requests: int = Field(
        default=30,
        alias="RATE_LIMIT_REQUESTS",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        alias="RATE_LIMIT_WINDOW_SECONDS",
    )

    # 为什么增加 embedding 并发数配置：
    # 批量给知识库做 embedding 时，真正的瓶颈通常不在本地 CPU，
    # 而在“同时对上游 embedding 服务发了多少请求”。
    # 做成配置后，可以按供应商限额灵活调整。
    embedding_max_concurrency: int = Field(
        default=5,
        alias="EMBEDDING_MAX_CONCURRENCY",
    )

    # 为什么增加缓存相关配置：
    # AI 系统里最贵、最慢的通常不是本地 Python 逻辑，
    # 而是外部模型调用。
    # 所以缓存的价值会非常直接：
    # - 命中一次，就少一次真实模型请求
    # - 成本下降
    # - 响应速度更快
    cache_prefix: str = Field(
        default="edu-agent:cache",
        alias="CACHE_PREFIX",
    )
    llm_cache_ttl_seconds: int = Field(
        default=600,
        alias="LLM_CACHE_TTL_SECONDS",
    )
    embedding_cache_ttl_seconds: int = Field(
        default=3600,
        alias="EMBEDDING_CACHE_TTL_SECONDS",
    )

    # 为什么增加 RAG chunk 配置：
    # 因为 chunk 大小和 overlap 会直接影响检索质量。
    # 做成配置后，后续调优会方便很多。
    rag_chunk_size: int = Field(
        default=500,
        alias="RAG_CHUNK_SIZE",
    )
    rag_chunk_overlap: int = Field(
        default=100,
        alias="RAG_CHUNK_OVERLAP",
    )

    # 为什么增加 top_k：
    # 因为 RAG 每次取回多少个相关 chunk，
    # 会影响上下文质量和提示词长度。
    rag_top_k: int = Field(
        default=3,
        alias="RAG_TOP_K",
    )

    # 为什么把 `.env` 配置也写进 model_config：
    # 虽然上面已经调用了 `load_dotenv()`，
    # 但这里再声明一遍能让 Settings 的来源更明确、更自描述。
    # 这样即使别人只看这个类，也能知道配置来自 `.env`。
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    这个函数的作用：
    返回一个全局复用的配置对象。

    为什么使用 `lru_cache`：
    因为配置通常在一个进程生命周期内不会频繁变化，
    所以没有必要每次调用都重新创建 `Settings` 实例。

    这样做了什么：
    第一次调用时会创建配置对象；
    后续再调用时，会直接复用第一次创建的结果。

    为什么不在全局直接写死一个 Settings 实例：
    使用函数包装更灵活，后续测试时也更容易替换或扩展。
    """
    return Settings()
