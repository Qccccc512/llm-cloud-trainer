import requests
import json
import time
import asyncio
import websockets

# 定义 Master 端模拟的请求数据
# 读取我们之前手动测试成功的配置
with open("cloud-llm/data/test_config.yaml", "r") as f:
    config_content = f.read()

# 为了避免和手动测试冲突，我们用一个新的 task_id，加上时间戳防止重名
task_id = f"task_verify_ws_{int(time.time())}"

# 构造 Payload
payload = {
    "task_id": task_id,
    "config_yaml": config_content
}

async def test_logs():
    print(f"Sending task {task_id} to Worker...")

    try:
        response = requests.post("http://localhost:8001/task/execute", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Task started. Connecting to WebSocket logs...")
            
            async with websockets.connect(f"ws://localhost:8001/task/{task_id}/log") as websocket:
                print("WebSocket Connected! Receiving logs:")
                print("-" * 50)
                try:
                    # 接收 20 秒日志后自动退出，避免测试无限挂起
                    while True:
                        message = await asyncio.wait_for(websocket.recv(), timeout=20.0)
                        print(f"[LOG] {message.strip()}")
                except asyncio.TimeoutError:
                    print("-" * 50)
                    print("Test finished (timeout reached).")
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_logs())