"""
这个文件的作用：
它是整个 FastAPI 服务的启动入口。

为什么入口文件要尽量清晰：
因为新同学第一次读一个后端项目时，通常最先看的就是入口文件。
如果入口文件过于复杂，会很难快速建立“项目是怎么跑起来的”整体认知。

这个文件主要负责三件事：
1. 读取全局配置
2. 创建 FastAPI 应用对象
3. 注册路由与基础接口

概念解释注释块：这个入口在 AI Agent 系统中的位置
--------------------------------------------------
对于 AI Agent 系统来说，`main.py` 并不负责真正的智能推理。
它更像是系统的大门和接线板：
- 大门：接收来自前端、客户端或其他服务的 HTTP 请求
- 接线板：把请求转交给 API 层、编排层、Agent 层等模块

也就是说：
`main.py` 负责“服务如何启动与暴露”，
而不是“Agent 如何思考与执行”。

为什么先保持简单：
真实生产项目里，这里可能还会加入：
- 中间件
- 异常处理器
- 日志初始化
- 生命周期管理
- 依赖注入容器

但对于教学级骨架，先把启动链路讲清楚，比一次塞进太多概念更重要。
--------------------------------------------------
"""

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.rate_limiter import SimpleRateLimitMiddleware

# 为什么先读取配置：
# 因为创建应用时，标题、环境名、第三方服务配置等信息可能都会用到。
# 先统一拿到配置，可以避免后面到处零散读取。
settings = get_settings()

# 为什么在模块级别创建 FastAPI 实例：
# 因为 Uvicorn 启动时会通过 `app.main:app` 找到这个对象。
# 这也是 FastAPI 项目最常见、最直观的组织方式。
app = FastAPI(
    # 标题会显示在 Swagger/OpenAPI 文档页中。
    title=settings.app_name,
    # 版本号方便后续接口迭代和部署追踪。
    version="0.1.0",
    # 描述信息会展示在接口文档中，帮助团队快速理解服务用途。
    description="面向 AI Agent 系统的教学级 FastAPI 项目骨架。",
)

# 为什么在入口层加中间件：
# 因为限流和并发保护属于“全局请求治理能力”。
# 它应该在业务逻辑真正执行前就生效，而不是散落在每个接口里分别处理。
app.add_middleware(
    SimpleRateLimitMiddleware,
    rate_limit_requests=settings.rate_limit_requests,
    rate_limit_window_seconds=settings.rate_limit_window_seconds,
    max_concurrent_requests=settings.max_concurrent_requests,
)

# 为什么使用 `include_router` 注册路由：
# 因为这样可以把不同领域的接口拆分到不同文件中。
# 例如未来可以有：
# - `app/api/chat_routes.py`
# - `app/api/admin_routes.py`
# - `app/api/knowledge_routes.py`
#
# `prefix="/api"` 的作用是给这组路由统一加上前缀，
# 所以 `routes.py` 里的 `/chat` 最终会变成 `/api/chat`。
app.include_router(api_router, prefix="/api")


@app.get("/", tags=["health"])
async def health_check() -> dict[str, str]:
    """
    这个函数的作用：
    提供一个最基础的健康检查接口。

    为什么后端服务通常都要有健康检查：
    1. 部署平台可以用它判断服务是否正常启动
    2. 运维或开发可以快速确认服务在线
    3. 排查问题时可以先区分“服务挂了”还是“业务逻辑有问题”

    这里做了什么：
    返回一个非常简单的 JSON，
    告诉调用方当前服务是可用状态，以及服务名是什么。
    """

    return {"status": "ok", "service": settings.app_name}
