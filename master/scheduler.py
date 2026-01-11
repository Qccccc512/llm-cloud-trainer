import asyncio
import aiohttp
import logging
from sqlmodel import Session, select
from datetime import datetime

from master.database import engine
from master.models import Task, TaskStatus, Worker, WorkerStatus
from master.utils import generate_config

logger = logging.getLogger("Scheduler")

async def dispatch_task(task: Task, worker: Worker, session: Session):
    """
    具体的任务分发逻辑
    """
    try:
        # 1. 生成 Config
        config_yaml = generate_config(task.id, task.config_json)
        
        # 2. 构造请求 Payload
        payload = {
            "task_id": task.id,
            "config_yaml": config_yaml
        }
        
        # 3. 发送 HTTP 请求给 Worker
        # 拼接 URL: http://1.2.3.4:8001/task/execute
        worker_url = worker.url.rstrip("/")
        target_url = f"{worker_url}/task/execute"
        
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(target_url, json=payload, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Task {task.id} dispatched to {worker.url}. Container: {data.get('container_id')}")
                    
                    # 更新状态
                    task.status = TaskStatus.RUNNING
                    task.worker_id = worker.id
                    task.updated_at = datetime.utcnow()
                    
                    worker.status = WorkerStatus.BUSY
                    worker.last_heartbeat = datetime.utcnow()
                    
                    session.add(task)
                    session.add(worker)
                    session.commit()
                    return True
                else:
                    text = await response.text()
                    logger.error(f"Worker refused task: {response.status} - {text}")
                    return False
                    
    except Exception as e:
        logger.error(f"Dispatch failed for task {task.id}: {e}")
        return False

async def scheduler_loop():
    """
    主调度循环
    """
    logger.info("Scheduler started.")
    while True:
        try:
            with Session(engine) as session:
                # 额外兜底：轮询 running 任务，检查是否已经结束，避免状态卡住
                await reconcile_running_tasks(session)
                await mark_offline_workers(session)

                # 1. 查找 Pending 任务 (FIFO)
                # 使用 select().limit(1) 避免并发冲突（简单起见）
                pending_task = session.exec(
                    select(Task).where(Task.status == TaskStatus.PENDING).order_by(Task.created_at)
                ).first()
                
                if pending_task:
                    # 2. 查找 Idle Worker
                    idle_worker = session.exec(
                        select(Worker).where(Worker.status == WorkerStatus.IDLE)
                    ).first()
                    
                    if idle_worker:
                        logger.info(f"Match found: Task {pending_task.id} -> Worker {idle_worker.url}")
                        # 发送任务 (异步)
                        await dispatch_task(pending_task, idle_worker, session)
                    else:
                        # 只有任务没有 Worker，等待
                        pass
                
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        
        # 间隔 3 秒
        await asyncio.sleep(3)


async def reconcile_running_tasks(session: Session):
    """Poll workers for running tasks to close out finished ones."""
    running_tasks = session.exec(select(Task).where(Task.status == TaskStatus.RUNNING)).all()
    if not running_tasks:
        return

    async with aiohttp.ClientSession() as http_session:
        for task in running_tasks:
            if not task.worker_id:
                continue
            worker = session.get(Worker, task.worker_id)
            if not worker:
                continue
            status_url = f"{worker.url.rstrip('/')}/task/status/{task.id}"
            try:
                async with http_session.get(status_url, timeout=5) as resp:
                    if resp.status != 200:
                        logger.error(f"Status check failed for task {task.id} ({resp.status})")
                        continue
                    data = await resp.json()
                    container_status = data.get("status")
                    exit_code = data.get("exit_code")

                    if container_status in {"exited", "dead", "not_found"}:
                        task.status = TaskStatus.SUCCESS if exit_code == 0 else TaskStatus.FAILED
                        task.updated_at = datetime.utcnow()
                        worker.status = WorkerStatus.IDLE
                        worker.last_heartbeat = datetime.utcnow()
                        session.add(task)
                        session.add(worker)
                        session.commit()
            except Exception as e:
                logger.error(f"Reconcile error for task {task.id}: {e}")


async def mark_offline_workers(session: Session, timeout_seconds: int = 60, purge_seconds: int = 24 * 3600):
    """将长时间无心跳的 Worker 标记为 offline，超长时间未恢复则清理注册。"""
    now = datetime.utcnow()
    workers = session.exec(select(Worker)).all()
    changed = False
    for w in workers:
        gap = (now - w.last_heartbeat).total_seconds()
        if gap > purge_seconds:
            # 中文注释：超过清理阈值直接删除注册，避免僵尸记录
            session.delete(w)
            changed = True
            continue
        if gap > timeout_seconds and w.status != WorkerStatus.OFFLINE:
            w.status = WorkerStatus.OFFLINE
            changed = True
            session.add(w)
    if changed:
        session.commit()
