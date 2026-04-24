"""
这个文件的作用：
实现 ToolAgent，也就是“工具 Agent”。

为什么多 Agent 架构里要单独有 ToolAgent：
因为“调用工具”本质上和“生成回答”不是同一类动作。

例如：
- 问时间
- 查天气
- 计算表达式
- 搜索文档

这些都更像“执行动作”而不是“自由生成文本”。

所以在架构上，把工具调用能力单独拆成 ToolAgent，
会比塞进 QAAgent 更清楚。

当前这个教学版先实现一个最简单工具：
- 时间查询

为什么只做这个：
因为它足够简单，能帮助理解 ToolAgent 的角色，
后续再逐步扩展到真正的搜索、数据库查询、外部 API 调用会更自然。
"""

from __future__ import annotations

from datetime import datetime

from app.agents.base_agent import AgentResult


class ToolAgent:
    """
    这个类的作用：
    负责执行工具型请求。
    """

    def __init__(self) -> None:
        self.agent_name = "tool_agent"

    async def run(self, user_message: str, session_id: str) -> AgentResult:
        """
        这个方法的作用：
        执行工具调用。

        当前版本里，如果进入 ToolAgent，
        我们就默认它是一个“时间查询”请求。
        """

        current_time = datetime.now().astimezone()
        answer = (
            "当前时间是："
            f"{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )

        trace = [
            {
                "step": "tool_execution",
                "content": "识别为时间查询，调用本地时间工具生成结果。",
            },
            {
                "step": "final",
                "content": answer,
            },
        ]

        return AgentResult(
            answer=answer,
            agent=self.agent_name,
            session_id=session_id,
            intent="tool",
            trace=trace,
            sources=[],
        )
