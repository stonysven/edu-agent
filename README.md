# edu-agent

`edu-agent` 是一个面向 AI Agent 系统的教学级 FastAPI 项目骨架。

这个项目当前还不是一个完整的智能体平台，而是一个“结构清晰、便于学习、方便后续扩展”的起点。
它的目标是帮助开发者先理解 AI Agent 系统在工程上的分层方式，再逐步往里面加入：

- Agent 执行逻辑
- RAG 检索增强生成
- Memory 记忆能力
- Tools 工具调用
- Orchestrator 编排流程

## 为什么这样分层

为了让系统更容易理解，我们把不同职责拆到了不同目录：

- `app/api`
  负责 HTTP 接口，也就是“服务如何接收请求、返回响应”
- `app/agents`
  负责智能体实现，也就是“谁来处理任务”
- `app/orchestrator`
  负责流程编排，也就是“一个请求要经历哪些步骤”
- `app/memory`
  负责记忆系统，也就是“系统记住了什么”
- `app/rag`
  负责检索增强，也就是“系统如何从外部知识中找答案”
- `app/tools`
  负责工具接入，也就是“系统如何调用外部能力”
- `app/core`
  负责基础能力，也就是“配置、常量、日志等全局能力”

这种分层方式的优点是：

1. 新人更容易找到代码位置
2. 不同模块职责更明确
3. 后续扩展不会把所有逻辑堆在一个文件里

## 项目结构

```text
edu-agent/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── agents/
│   │   └── __init__.py
│   ├── orchestrator/
│   │   └── __init__.py
│   ├── memory/
│   │   └── __init__.py
│   ├── tools/
│   │   └── __init__.py
│   ├── rag/
│   │   └── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   └── __init__.py
├── Makefile
├── .env.example
├── requirements.txt
└── README.md
```

## 当前已实现功能

- `GET /`
  提供健康检查接口，用来确认服务是否正常启动
- `POST /api/chat`
  提供最小可运行的 Agent 聊天接口，调用 LLM 返回真实结果
- `.env` 配置加载
  当前已支持读取 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`

## 为什么当前实现保持简单

这个骨架没有一开始就加入复杂的 Agent 调度、向量数据库、记忆召回和工具注册系统，
原因不是这些不重要，而是教学阶段更应该先把基础结构搭稳。

如果一开始引入太多复杂概念，会带来这些问题：

- 新手难以快速跑通项目
- 出错时不容易定位问题
- 还没理解边界就过度抽象，后续容易返工

所以目前采用的是“先简单跑通，再逐步增强”的方案。

## 一键启动步骤（从 0 开始）

下面的方式适用于 Mac 和 Linux。

### 方式一：推荐，用 Makefile，2 步启动

第 1 步：进入项目目录

```bash
cd edu-agent
```

第 2 步：一键安装并启动

```bash
make dev
```

这个命令会自动完成：

1. 创建虚拟环境 `.venv`
2. 升级 `pip`
3. 安装 `requirements.txt` 中的依赖
4. 启动 FastAPI 服务

### 方式二：分步执行，最多 3 步

如果你希望更清楚地看到每一步在做什么，可以使用下面的命令：

第 1 步：进入项目目录

```bash
cd edu-agent
```

第 2 步：准备运行环境

```bash
make setup
```

第 3 步：启动服务

```bash
make run
```

## 环境变量说明

项目提供了 `.env.example`，建议先复制出一份本地 `.env`：

```bash
cp .env.example .env
```

然后按需编辑 `.env` 文件：

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=60
```

当前骨架阶段即使不填真实 Key，项目也可以启动。
这样做是为了降低学习门槛，方便先熟悉工程结构。

## 如果系统没有 make

极少数环境可能没有预装 `make`。这种情况下也可以手动执行：

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

## 服务启动后访问地址

- 健康检查：[http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Swagger 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 接口示例

请求：

```bash
curl -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"你好，Agent"}'
```

响应：

```json
{
  "answer": "这里会返回模型生成的真实回答",
  "agent": "simple_agent",
  "trace": [
    {
      "step": "thought",
      "content": "用户正在咨询：你好，Agent"
    },
    {
      "step": "llm_call",
      "content": "调用 LLM 生成回答，模型为 gpt-4o-mini"
    },
    {
      "step": "final",
      "content": "这里会记录最终答案"
    }
  ]
}
```

## 后续建议扩展顺序

如果你打算继续把这个骨架演进成真正的 AI Agent 系统，建议按下面顺序逐步实现：

1. 在 `agents` 中实现一个最小 Agent 服务类
2. 在 `orchestrator` 中加入简单的请求编排逻辑
3. 在 `memory` 中加入会话级短期记忆
4. 在 `rag` 中加入最小检索链路
5. 在 `tools` 中接入一个简单外部工具

这样推进的好处是：
每一步都能独立验证，不容易一下子把系统做得过于复杂。

## Makefile 命令说明

```bash
make help
```

你还可以使用这些常见命令：

- `make venv`
  只创建虚拟环境
- `make install`
  创建虚拟环境并安装依赖
- `make setup`
  一次完成环境准备
- `make run`
  启动服务
- `make dev`
  一键安装并启动
- `make clean`
  删除虚拟环境，便于重建
