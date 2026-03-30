# 为什么使用 Makefile：
# 因为它是 Mac/Linux 上非常常见的任务入口文件，
# 能把“创建虚拟环境、安装依赖、启动服务”这些重复命令收敛成更短的指令。
#
# 为什么这里优先选择 Makefile，而不是更复杂的脚本系统：
# 1. 学习成本低，新手容易看懂
# 2. Mac/Linux 默认通常可用，兼容性较好
# 3. 对当前教学级项目来说已经足够，不需要引入更重的任务管理工具

# 为什么显式指定 SHELL：
# 这样可以让 Make 在执行命令时使用统一的 shell，
# 减少不同环境下行为不一致的问题。
SHELL := /bin/bash

# 为什么把这些变量提出来：
# 因为后续如果要修改 Python 版本命令、虚拟环境目录、启动端口，
# 只需要改一处，不需要全局搜索替换。
PYTHON := python3
VENV_DIR := .venv
PIP := $(VENV_DIR)/bin/pip
UVICORN := $(VENV_DIR)/bin/uvicorn
APP := app.main:app
HOST := 127.0.0.1
PORT := 8000

# 为什么把这些目标声明为 .PHONY：
# 因为这些目标表示“动作”，不是“生成某个同名文件”。
# 这样可以避免目录里如果刚好有同名文件时，Make 错误地跳过执行。
.PHONY: help venv install setup run dev stop clean

help:
	@echo "可用命令："
	@echo "  make venv    - 创建 Python 虚拟环境"
	@echo "  make install - 安装项目依赖"
	@echo "  make setup   - 一次完成创建虚拟环境 + 安装依赖"
	@echo "  make run     - 启动 FastAPI 服务"
	@echo "  make dev     - 一键安装并启动服务"
	@echo "  make stop    - 停止当前占用 8000 端口的 uvicorn 进程"
	@echo "  make clean   - 删除虚拟环境"

venv:
	# 为什么先判断目录是否存在：
	# 这样可以避免每次执行都重复创建虚拟环境。
	# 如果虚拟环境已经存在，我们直接复用即可。
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "已创建虚拟环境：$(VENV_DIR)"; \
	else \
		echo "虚拟环境已存在：$(VENV_DIR)"; \
	fi

install: venv
	# 为什么先升级 pip：
	# 新环境里的 pip 版本可能较旧，升级后通常能减少依赖安装问题。
	$(PIP) install --upgrade pip
	# 这里做了什么：
	# 使用 requirements.txt 中声明的依赖安装项目运行所需的库。
	$(PIP) install -r requirements.txt

setup: install
	@echo "环境准备完成，可以执行 make run 启动服务。"

run:
	# 为什么直接使用虚拟环境中的 uvicorn：
	# 这样可以确保启动时使用的是当前项目安装的依赖版本，
	# 避免误用系统全局环境里的包。
	#
	# 为什么额外加 `--env-file .env`：
	# 虽然应用代码里已经会读取 `.env`，
	# 但在 uvicorn 的 reload 多进程场景下，显式指定环境文件会更稳妥、更直观。
	$(UVICORN) $(APP) --reload --host $(HOST) --port $(PORT) --env-file .env

dev: install run

stop:
	# 为什么提供 stop：
	# 很多“明明改了配置却不生效”的问题，本质上是旧的 uvicorn 进程还占着端口。
	# 这个命令会清理当前项目常见的 uvicorn 进程，方便重新启动。
	-pkill -f "uvicorn app.main:app" || true

clean:
	# 为什么提供 clean：
	# 当依赖环境损坏，或者你希望彻底重建环境时，
	# 删除虚拟环境再重新安装通常是最简单直接的办法。
	rm -rf $(VENV_DIR)
	@echo "已删除虚拟环境：$(VENV_DIR)"
