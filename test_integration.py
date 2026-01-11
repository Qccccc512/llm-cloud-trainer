import requests
import time
import sys

MASTER_URL = "http://localhost:8000"
WORKER_URL = "http://localhost:8001" 

def test_integration():
    print(">>> 1. Registering Worker...")
    try:
        # 手动注册 Worker
        resp = requests.post(f"{MASTER_URL}/worker/register", json={"url": WORKER_URL})
        print(f"Register Response: {resp.json()}")
    except Exception as e:
        print(f"Failed to register worker: {e}")
        sys.exit(1)

    print("\n>>> 2. Submitting Task...")
    try:
        # 提交一个简单的任务
        payload = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "epochs": 1.0, # 跑少一点，为了快速看效果
            "batch_size": 2,
            "learning_rate": 2e-4
        }
        resp = requests.post(f"{MASTER_URL}/submit", json=payload)
        task_data = resp.json()
        task_id = task_data["id"]
        print(f"Task Submitted: ID={task_id}, Status={task_data['status']}")
    except Exception as e:
        print(f"Failed to submit task: {e}")
        sys.exit(1)

    print("\n>>> 3. Waiting for Scheduler (Polling Task Status)...")
    for i in range(10):
        try:
            resp = requests.get(f"{MASTER_URL}/tasks")
            tasks = resp.json()
            # 找到我们的任务
            my_task = next((t for t in tasks if t["id"] == task_id), None)
            
            if my_task:
                print(f"[{i}s] Task Status: {my_task['status']}")
                if my_task['status'] == 'running':
                    print("\nSUCCESS! Task is RUNNING. Scheduler works!")
                    return
            
            time.sleep(1)
        except Exception as e:
            print(f"Error polling: {e}")
    
    print("\nFAILED: Task did not turn to RUNNING after 10 seconds.")

if __name__ == "__main__":
    test_integration()
