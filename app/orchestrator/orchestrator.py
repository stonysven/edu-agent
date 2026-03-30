"""
这个文件的作用：
定义当前项目的最小编排器（Orchestrator）。

为什么需要 Orchestrator：
虽然当前版本只有一个 `simple_agent`，
但从工程分层角度看，接口层最好不要直接依赖具体 Agent。

更合理的职责分工是：
- API 层：接收请求、返回响应
- Orchestrator 层：决定调用哪个 Agent
- Agent 层：真正执行智能逻辑

这样做的好处是：
1. API 层更干净
2. 后续增加多 Agent 时改动更小
3. 系统边界更清晰

概念解释注释块：Orchestrator 的角色
--------------------------------------------------
Orchestrator 可以理解为“调度员”或“编排器”。

它不一定亲自完成任务，
但它负责决定：
- 这个请求应该交给谁处理
- 是否要先查记忆
- 是否要先做 RAG
- 是否需要多 Agent 协作

当前最小版本里，
它只做一件事：
把用户输入交给 `simple_agent`。

为什么即便只有一个 Agent 也要先加编排器：
因为这是一个很关键的工程边界。
提前把边界划出来，后续扩展会轻松很多。
--------------------------------------------------
"""

from app.agents.base_agent import AgentResult
from app.agents.simple_agent import SimpleAgent


class Orchestrator:
    """
    这个类的作用：
    作为当前系统的最小请求编排入口。

    为什么在初始化时创建 `SimpleAgent`：
    因为当前只有一个 Agent，直接持有它最简单直观。
    后续如果 Agent 变多，可以把这里扩展成路由分发逻辑。
    """

    def __init__(self) -> None:
        self.simple_agent = SimpleAgent()

    def handle_chat(self, user_message: str) -> AgentResult:
        """
        这个方法的作用：
        接收用户输入，并将其转发给当前负责处理聊天任务的 Agent。

        为什么单独提供 `handle_chat`：
        因为编排器对外暴露的应该是“业务动作”，
        而不是内部某个 Agent 的具体方法名。

        现在做了什么：
        1. 接收用户消息
        2. 调用 `simple_agent.run()`
        3. 返回统一的 `AgentResult`
        """

        return self.simple_agent.run(user_message=user_message)
