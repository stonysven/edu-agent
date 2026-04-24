"""
这个文件的作用：
提供一个最小可用的并发请求测试脚本。

为什么需要它：
并发优化不是“看代码感觉应该更快”就算完成，
而是要能实际发一批并发请求，看服务在高并发场景下的表现。

这个脚本会做什么：
1. 并发发送多次 `/api/chat` 请求
2. 统计成功数、失败数和总耗时
3. 帮助观察：
   - 服务是否还能响应
   - 是否触发限流
   - 平均延迟大致是多少

为什么这里使用 `httpx.AsyncClient`：
因为我们就是要测试“并发请求”。
异步客户端能更直接地模拟多个请求同时打到服务的情况。
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx


async def send_chat_request(
    client: httpx.AsyncClient,
    url: str,
    message: str,
    session_prefix: str,
    index: int,
) -> dict[str, object]:
    """
    这个函数的作用：
    发送一次 `/api/chat` 请求，并记录结果。
    """

    started_at = time.perf_counter()
    try:
        response = await client.post(
            url,
            json={
                "message": message,
                "session_id": f"{session_prefix}-{index}",
            },
        )
        latency_seconds = time.perf_counter() - started_at
        return {
            "index": index,
            "status_code": response.status_code,
            "latency_seconds": latency_seconds,
        }
    except httpx.HTTPError as exc:
        latency_seconds = time.perf_counter() - started_at
        return {
            "index": index,
            "status_code": 0,
            "latency_seconds": latency_seconds,
            "error": str(exc),
        }


async def main() -> None:
    """
    这个函数的作用：
    解析命令行参数并发起一轮并发测试。
    """

    parser = argparse.ArgumentParser(description="并发测试 /api/chat 接口")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/api/chat",
        help="要压测的目标接口地址",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="同时并发的请求数量",
    )
    parser.add_argument(
        "--message",
        default="你好，请简单介绍一下AI Agent。",
        help="发送给接口的测试消息",
    )
    parser.add_argument(
        "--session-prefix",
        default="load-test",
        help="测试会话 ID 前缀",
    )
    args = parser.parse_args()

    started_at = time.perf_counter()

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            send_chat_request(
                client=client,
                url=args.url,
                message=args.message,
                session_prefix=args.session_prefix,
                index=index,
            )
            for index in range(args.concurrency)
        ]
        results = await asyncio.gather(*tasks)

    total_elapsed_seconds = time.perf_counter() - started_at
    success_count = sum(1 for item in results if item["status_code"] == 200)
    limited_count = sum(1 for item in results if item["status_code"] == 429)
    failure_count = len(results) - success_count - limited_count
    average_latency = (
        sum(float(item["latency_seconds"]) for item in results) / len(results)
        if results
        else 0.0
    )

    print("并发测试完成")
    print(f"目标接口: {args.url}")
    print(f"并发数: {args.concurrency}")
    print(f"成功数: {success_count}")
    print(f"限流数: {limited_count}")
    print(f"失败数: {failure_count}")
    print(f"平均延迟(秒): {average_latency:.3f}")
    print(f"总耗时(秒): {total_elapsed_seconds:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
