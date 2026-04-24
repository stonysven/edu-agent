"""
这个文件的作用：
定义对外提供的 HTTP 路由。

当前这个文件同时承担两类接口：
1. `/chat`：多 Agent 对话入口
2. `/upload` + `/upload-file` + `/ask`：原始手写版 RAG 知识库加载与问答接口
3. `/upload-langchain` + `/upload-file-langchain` + `/ask-langchain`：LangChain 版 RAG 接口

为什么 `/chat` 要升级成多 Agent 入口：
因为现在系统已经不是“单一 Agent 直接回答”，
而是：
1. 先做规划
2. 再决定走 chat / rag / tool
3. 再由具体 Agent 执行
"""

import asyncio
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.agents.langchain_rag_agent import LangChainRAGAgent
from app.core.cache_manager import get_cache_manager
from app.orchestrator.orchestrator import Orchestrator

router = APIRouter(tags=["chat", "rag"])
orchestrator = Orchestrator()
rag_agent = orchestrator.qa_agent.rag_agent
langchain_rag_agent = LangChainRAGAgent()
cache_manager = get_cache_manager()


class ChatRequest(BaseModel):
    """
    这个类的作用：
    定义 `/chat` 接口的请求体结构。
    """

    message: str = Field(..., description="用户输入的消息内容")
    session_id: str | None = Field(
        default=None,
        description="当前对话会话 ID；如果不传，服务端会自动生成",
    )


class ChatResponse(BaseModel):
    """
    这个类的作用：
    定义 `/chat` 接口的标准返回结构。

    字段说明：
    - answer：最终回答
    - agent：最终执行的 Agent
    - intent：规划阶段判断出的意图
    - session_id：当前会话 ID
    - trace：完整执行轨迹
    - sources：如果走 RAG，可返回引用来源
    """

    answer: str
    agent: str
    intent: str
    session_id: str
    trace: list[Any]
    sources: list[dict[str, Any]] = Field(default_factory=list)


class UploadKnowledgeRequest(BaseModel):
    directory: str = Field(
        default="data",
        description="本地知识库目录，默认读取项目根目录下的 data/",
    )


class UploadKnowledgeResponse(BaseModel):
    status: str
    agent: str
    directory: str
    document_count: int
    chunk_count: int


class UploadFileResponse(BaseModel):
    status: str
    agent: str
    files: list[str]
    document_count: int
    chunk_count: int


class AskRequest(BaseModel):
    question: str = Field(..., description="用户要基于知识库提问的问题")


class AskResponse(BaseModel):
    answer: str
    agent: str
    sources: list[dict[str, Any]]
    trace: list[Any]


class CompareRAGResult(BaseModel):
    """
    这个类的作用：
    描述单个 RAG 实现的对比结果。
    """

    answer: str
    agent: str
    sources: list[dict[str, Any]]
    trace: list[Any]


class CompareRAGResponse(BaseModel):
    """
    这个类的作用：
    统一返回“原始 RAG”和“LangChain RAG”的对比结果。

    为什么这里用一个独立响应模型：
    因为对比接口的重点不是单个答案，
    而是“同一个问题在两套实现下分别得到了什么结果”。
    这样前端和调用方就可以一次拿到两边完整输出，直接做对照。
    """

    question: str
    original_rag: CompareRAGResult
    langchain_rag: CompareRAGResult


class CacheStatsResponse(BaseModel):
    """
    这个类的作用：
    定义缓存统计接口的返回结构。
    """

    backend: str
    hit_count: int
    miss_count: int


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    这个函数的作用：
    处理多 Agent 对话请求。
    """

    try:
        result = await orchestrator.handle_chat(
            user_message=request.message,
            session_id=request.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        agent=result.agent,
        intent=result.intent,
        session_id=result.session_id,
        trace=result.trace,
        sources=result.sources,
    )


@router.post("/upload", response_model=UploadKnowledgeResponse)
async def upload_knowledge(request: UploadKnowledgeRequest) -> UploadKnowledgeResponse:
    try:
        result = await rag_agent.load_knowledge_base(directory=request.directory)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadKnowledgeResponse(
        status="success",
        agent=rag_agent.agent_name,
        directory=str(result["directory"]),
        document_count=int(result["document_count"]),
        chunk_count=int(result["chunk_count"]),
    )


@router.post("/upload-langchain", response_model=UploadKnowledgeResponse)
async def upload_knowledge_langchain(
    request: UploadKnowledgeRequest,
) -> UploadKnowledgeResponse:
    """
    这个函数的作用：
    为 LangChain 版本的 RAG 构建向量库。
    """

    try:
        result = await langchain_rag_agent.load_knowledge_base(directory=request.directory)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadKnowledgeResponse(
        status="success",
        agent=langchain_rag_agent.agent_name,
        directory=str(result["directory"]),
        document_count=int(result["document_count"]),
        chunk_count=int(result["chunk_count"]),
    )


@router.post("/upload-file", response_model=UploadFileResponse)
async def upload_file(files: list[UploadFile] = File(...)) -> UploadFileResponse:
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件。")

    uploaded_files: list[tuple[str, bytes]] = []
    file_names: list[str] = []

    for file in files:
        file_names.append(file.filename)
        file_bytes = await file.read()
        uploaded_files.append((file.filename, file_bytes))

    try:
        result = await rag_agent.load_uploaded_files(files=uploaded_files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadFileResponse(
        status="success",
        agent=rag_agent.agent_name,
        files=file_names,
        document_count=int(result["document_count"]),
        chunk_count=int(result["chunk_count"]),
    )


@router.post("/upload-file-langchain", response_model=UploadFileResponse)
async def upload_file_langchain(files: list[UploadFile] = File(...)) -> UploadFileResponse:
    """
    这个函数的作用：
    为 LangChain 版本的 RAG 处理上传文件入库。
    """

    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件。")

    uploaded_files: list[tuple[str, bytes]] = []
    file_names: list[str] = []

    for file in files:
        file_names.append(file.filename)
        file_bytes = await file.read()
        uploaded_files.append((file.filename, file_bytes))

    try:
        result = await langchain_rag_agent.load_uploaded_files(files=uploaded_files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadFileResponse(
        status="success",
        agent=langchain_rag_agent.agent_name,
        files=file_names,
        document_count=int(result["document_count"]),
        chunk_count=int(result["chunk_count"]),
    )


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        result = await rag_agent.ask(question=request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(
        answer=result["answer"],
        agent=result["agent"],
        sources=result["sources"],
        trace=result["trace"],
    )


@router.post("/ask-langchain", response_model=AskResponse)
async def ask_langchain(request: AskRequest) -> AskResponse:
    """
    这个函数的作用：
    执行一次 LangChain 版本的 RAG 问答。
    """

    try:
        result = await langchain_rag_agent.ask(question=request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(
        answer=result["answer"],
        agent=result["agent"],
        sources=result["sources"],
        trace=result["trace"],
    )


@router.post("/compare-rag", response_model=CompareRAGResponse)
async def compare_rag(request: AskRequest) -> CompareRAGResponse:
    """
    这个函数的作用：
    对同一个问题同时调用“原始手写版 RAG”和“LangChain 版 RAG”，
    并把两边结果一次性返回。

    为什么这个接口很有价值：
    因为它让“教学版底层实现”和“工业版封装实现”不只是代码并存，
    还真正具备了可直接 A/B 对比的调用入口。
    """

    try:
        original_result, langchain_result = await asyncio.gather(
            rag_agent.ask(question=request.question),
            langchain_rag_agent.ask(question=request.question),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CompareRAGResponse(
        question=request.question,
        original_rag=CompareRAGResult(
            answer=original_result["answer"],
            agent=original_result["agent"],
            sources=original_result["sources"],
            trace=original_result["trace"],
        ),
        langchain_rag=CompareRAGResult(
            answer=langchain_result["answer"],
            agent=langchain_result["agent"],
            sources=langchain_result["sources"],
            trace=langchain_result["trace"],
        ),
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats() -> CacheStatsResponse:
    """
    这个函数的作用：
    返回当前缓存命中统计。

    为什么需要这个接口：
    因为做了缓存之后，我们不能只“感觉可能更快了”，
    还应该能看到缓存到底命中了多少次、未命中了多少次。
    """

    stats = await cache_manager.get_stats()
    return CacheStatsResponse(
        backend=str(stats["backend"]),
        hit_count=int(stats["hit_count"]),
        miss_count=int(stats["miss_count"]),
    )
