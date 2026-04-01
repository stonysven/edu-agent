"""
这个文件的作用：
定义所有 Agent 的基础接口与统一结果结构。

为什么需要一个 `base_agent.py`：
在 AI Agent 系统里，不同 Agent 的能力可能不同，
但它们通常都会遵守一套相似的输入输出协议。

例如：
- 都需要接收用户输入
- 都需要知道这条消息属于哪个 session
- 都需要返回答案
- 都需要返回执行轨迹 trace
- 在多 Agent 架构里，通常还会带上 intent 和 sources

如果没有一个基础约定，后续一旦增加多个 Agent，
接口层和编排层就会很难统一调用它们。

为什么这里选择“简单基类 + 数据类”：
1. 新手容易理解
2. 后续容易扩展
3. 不需要一开始就引入复杂框架

概念解释注释块：Agent 在系统中的角色
--------------------------------------------------
Agent 可以理解为“负责完成某类任务的智能执行单元”。

它和普通函数的区别在于：
1. Agent 往往会先理解目标
2. 再决定用什么方式完成任务
3. 过程中可能调用 LLM、工具、记忆或检索模块
4. 最后返回结果和过程轨迹

在这个项目的第一版中，
`simple_agent` 只是一个最小实现：
- 输入：用户消息
- 动作：调用一次 LLM
- 输出：答案 + trace

后续如果演进成多 Agent 系统，
我们可以继续新增：
- PlannerAgent
- TutorAgent
- SearchAgent
- MemoryAgent
等等
--------------------------------------------------
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentResult:
    """
    这个数据类的作用：
    统一描述 Agent 执行完成后的返回结果。

    为什么要单独定义一个结果对象：
    因为后续无论是 API 层还是 Orchestrator 层，
    都希望拿到一个结构稳定的结果。

    这里包含六部分核心信息：
    - answer：给用户的最终回答
    - agent：本次负责处理请求的 Agent 名称
    - session_id：本次对话所属会话 ID
    - intent：本次请求最终走的是哪条处理路径
    - trace：本次执行过程中的关键步骤
    - sources：如果走的是 RAG，可返回引用来源
    """

    answer: str
    agent: str
    session_id: str
    intent: str
    trace: list[dict[str, str]] = field(default_factory=list)
    sources: list[dict[str, object]] = field(default_factory=list)


class BaseAgent(ABC):
    """
    这个抽象基类的作用：
    定义所有 Agent 都应该具备的最基本行为。

    为什么使用抽象基类：
    因为我们希望后续每个 Agent 至少都实现 `run()` 方法。
    这样 Orchestrator 在调用不同 Agent 时，就能使用统一方式。

    为什么这里不设计得更复杂：
    当前阶段只需要一个“最小但清晰”的约定即可。
    如果现在就加入复杂状态机、工具注册系统、插件机制，
    反而会让新手更难理解。
    """

    @abstractmethod
    def run(self, user_message: str, session_id: str) -> AgentResult:
        """
        这个方法的作用：
        定义 Agent 的统一执行入口。

        参数说明：
        - `user_message`：用户输入的文本消息
        - `session_id`：当前对话所属会话 ID

        返回说明：
        - 返回 `AgentResult`，其中包含最终回答和执行轨迹
        """

        raise NotImplementedError
