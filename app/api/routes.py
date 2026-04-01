"""
这个文件的作用：
定义对外提供的 HTTP 路由。

当前这个文件同时承担两类接口：
1. `/chat`：多 Agent 对话入口
2. `/upload` + `/upload-file` + `/ask`：RAG 知识库加载与问答接口

为什么 `/chat` 要升级成多 Agent 入口：
因为现在系统已经不是“单一 Agent 直接回答”，
而是：
1. 先做规划
2. 再决定走 chat / rag / tool
3. 再由具体 Agent 执行
"""

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.agents.rag_agent import RAGAgent
from app.orchestrator.orchestrator import Orchestrator

router = APIRouter(tags=["chat", "rag"])
orchestrator = Orchestrator()
rag_agent = orchestrator.qa_agent.rag_agent


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


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    这个函数的作用：
    处理多 Agent 对话请求。
    """

    try:
        result = orchestrator.handle_chat(
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
        result = rag_agent.load_knowledge_base(directory=request.directory)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadKnowledgeResponse(
        status="success",
        agent=rag_agent.agent_name,
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
        result = rag_agent.load_uploaded_files(files=uploaded_files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadFileResponse(
        status="success",
        agent=rag_agent.agent_name,
        files=file_names,
        document_count=int(result["document_count"]),
        chunk_count=int(result["chunk_count"]),
    )


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        result = rag_agent.ask(question=request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(
        answer=result["answer"],
        agent=result["agent"],
        sources=result["sources"],
        trace=result["trace"],
    )
