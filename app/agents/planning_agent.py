"""
这个文件的作用：
实现 PlanningAgent，也就是“规划 Agent”。

为什么多 Agent 系统通常要先有 PlanningAgent：
因为当系统里不止一种处理方式时，
第一步通常不是“立刻回答”，而是“先判断该走哪条路”。

例如：
- 普通闲聊，直接走 chat
- 基于知识库的问题，走 rag
- 明确的工具型请求，走 tool

这一步就叫 Planning。

概念解释注释块：为什么要先做 Planning
--------------------------------------------------
如果不先做 Planning，而是直接让一个 Agent 处理所有事情，
会遇到这些问题：
1. 逻辑会越来越混乱
2. 不同能力之间边界不清
3. 后续新增 Agent 时很难扩展

所以多 Agent 架构里常见的模式是：
1. 先判断任务类型
2. 再选执行 Agent

这也是为什么 PlanningAgent 通常是编排流程的第一站。
--------------------------------------------------

概念解释注释块：为什么不能直接让 LLM 处理所有任务
--------------------------------------------------
从表面看，好像“让一个大模型全处理”最简单。
但工程上并不总是最优。

原因包括：
1. 普通聊天、RAG、工具调用，本质上是不同任务
2. 它们的输入结构、输出结构和可靠性要求不同
3. 全塞给一个 Agent，后续维护和调试会变困难

所以更合理的做法是：
- PlanningAgent 负责判断
- QAAgent 负责回答
- ToolAgent 负责工具执行

这样系统的职责会更清楚，也更容易扩展。
--------------------------------------------------
"""

from __future__ import annotations


PlanningResult = dict[str, str]


class PlanningAgent:
    """
    这个类的作用：
    负责分析用户问题类型，并给出路由建议。

    当前这个教学版采用的是“规则判断”而不是“再调用一次 LLM 判断”。

    为什么先用规则判断：
    1. 更容易理解
    2. 更稳定
    3. 不增加额外模型调用成本

    后续如果系统变复杂，可以再升级成：
    - 规则 + LLM 混合规划
    - 多阶段规划
    - 带置信度的路由判断
    """

    TOOL_KEYWORDS = [
        "现在几点",
        "几点了",
        "当前时间",
        "今天几号",
        "日期",
        "时间",
        "time",
        "date",
    ]

    RAG_KEYWORDS = [
        "知识库",
        "根据文档",
        "根据资料",
        "根据知识",
        "上传的文件",
        "课程内容",
        "本地文档",
        "样例文档",
        "sample_knowledge",
        "rag",
    ]

    async def plan(self, user_message: str, rag_available: bool) -> PlanningResult:
        """
        这个方法的作用：
        根据用户输入和当前系统状态，判断应该走哪种处理路径。

        返回结构说明：
        - `intent`：chat / rag / tool
        - `reason`：为什么这么判断
        """

        normalized_message = user_message.lower().strip()

        if any(keyword in normalized_message for keyword in self.TOOL_KEYWORDS):
            return {
                "intent": "tool",
                "reason": "问题包含时间/日期类关键词，适合交给 ToolAgent 处理。",
            }

        if rag_available and any(keyword.lower() in normalized_message for keyword in self.RAG_KEYWORDS):
            return {
                "intent": "rag",
                "reason": "问题明确提到知识库/文档/资料，且当前已加载知识库，适合走 RAG。",
            }

        return {
            "intent": "chat",
            "reason": "问题更像普通聊天或通用问答，先走 QAAgent 的普通聊天模式。",
        }
