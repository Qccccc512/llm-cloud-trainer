from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
import docker
import os
import asyncio
import logging
import aiohttp
import datetime
import subprocess
from typing import Dict
import yaml
try:
    import psutil
except ImportError:  # 中文注释：若未安装 psutil，资源监控降级为 0
    psutil = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Worker")

app = FastAPI()

# 全局 Docker 客户端
client = docker.from_env()


def load_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_ipinfo(ipinfo_path: str) -> dict:
    """支持 yaml 或 key=value 文本，返回键值字典。"""
    if not os.path.isfile(ipinfo_path):
        return {}
    try:
        with open(ipinfo_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v is not None}
    except Exception:
        pass

    result = {}
    try:
        with open(ipinfo_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if key:
                    result[key] = val
    except Exception:
        return {}
    return result


def choose_url(env_key: str, conf_key: str, ipinfo_key_private: str, ipinfo_key_public: str, default: str, conf: dict, ipinfo: dict) -> str:
    return (
        os.environ.get(env_key)
        or conf.get(conf_key)
        or ipinfo.get(ipinfo_key_private)
        or ipinfo.get(ipinfo_key_public)
        or default
    )


WORKER_CONFIG_PATH = os.environ.get(
    "WORKER_CONFIG", os.path.join(os.path.dirname(__file__), "config", "config.yaml")
)
CONF = load_config(WORKER_CONFIG_PATH)
IPINFO_PATH = os.environ.get(
    "IPINFO_PATH", os.path.join(os.path.dirname(__file__), "master-worker-ipinfo")
)
IPINFO = load_ipinfo(IPINFO_PATH)

# 共享目录的绝对路径：默认指向项目下 cloud-llm，可通过环境变量或配置覆写，避免硬编码路径
HOST_BASE_PATH = os.environ.get(
    "HOST_BASE_PATH",
    CONF.get(
        "host_base_path",
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "cloud-llm")),
    ),
)
LOG_BASE_PATH = os.environ.get(
    "LOG_BASE_PATH", CONF.get("log_base_path", os.path.join(HOST_BASE_PATH, "logs"))
)
HF_CACHE_PATH = os.environ.get(
    "HF_CACHE_PATH", CONF.get("hf_cache_path", os.path.join(HOST_BASE_PATH, "hf-cache"))
)

MASTER_URL = choose_url(
    "MASTER_URL",
    "master_url",
    "master_private_url",
    "master_public_url",
    "http://localhost:8000",
    CONF,
    IPINFO,
)
WORKER_URL = choose_url(
    "WORKER_URL",
    "worker_url",
    "worker_private_url",
    "worker_public_url",
    "http://localhost:8001",
    CONF,
    IPINFO,
)
ADVERTISE_URL = os.environ.get(
    "WORKER_ADVERTISE_URL",
    CONF.get("worker_advertise_url")
    or IPINFO.get("worker_private_url")
    or WORKER_URL,
)

os.makedirs(HF_CACHE_PATH, exist_ok=True)
os.makedirs(LOG_BASE_PATH, exist_ok=True)
logger.info(f"HOST_BASE_PATH set to: {HOST_BASE_PATH}")
logger.info(f"MASTER_URL set to: {MASTER_URL}")
logger.info(f"WORKER_URL set to: {WORKER_URL}")
logger.info(f"ADVERTISE_URL set to: {ADVERTISE_URL}")

# 记录用户推理会话：user_id -> {container_id, port, last_used}
inference_sessions: Dict[int, Dict] = {}
inference_lock = asyncio.Lock()

async def cleanup_loop():
    """后台定时清理退出的容器（带延迟，避免过早删除）。"""
    while True:
        try:
            # 中文注释：统一清理所有本服务拉起且已退出的容器（训练/评估/导出/推理），避免残留
            containers = client.containers.list(
                all=True,
                filters={"status": "exited", "name": "llm-"}
            )
            now = datetime.datetime.utcnow()
            for container in containers:
                try:
                    container.reload()
                    finished = container.attrs.get("State", {}).get("FinishedAt")
                    if finished and finished != "0001-01-01T00:00:00Z":
                        try:
                            finished_dt = datetime.datetime.fromisoformat(finished.replace("Z", "+00:00"))
                            delta = (now - finished_dt).total_seconds()
                            # 延迟 300 秒后再清理，避免抢先删除导致 Master 未拿到状态
                            if delta < 300:
                                continue
                        except Exception:
                            pass
                    logger.info(f"Removing exited container: {container.name} ({container.id[:12]})")
                    container.remove()
                except Exception as e:
                    logger.error(f"Failed to remove container {container.name}: {e}")
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")

        await asyncio.sleep(30)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_loop())
    asyncio.create_task(inference_reaper())
    # 确保已注册后再开始心跳上报
    await register_self()
    asyncio.create_task(heartbeat_loop())

class TrainTask(BaseModel):
    task_id: str
    config_yaml: str  # 接收完整的 yaml 内容


class EvalTask(BaseModel):
    eval_id: str
    config_yaml: str


class ExportTask(BaseModel):
    export_id: str
    config_yaml: str


class InferenceStart(BaseModel):
    deploy_id: str | None = None
    user_id: int
    config_yaml: str | None = None
    api_port: int | None = None
    api_host: str | None = None
    api_key: str | None = None
    api_model_name: str | None = None
    api_verbose: str | None = None
    max_concurrent: int | None = None
    fastapi_root_path: str | None = None
    # 兼容旧字段
    task_id: str | None = None
    model_path: str | None = None


class InferenceStop(BaseModel):
    user_id: int


class InferenceChat(BaseModel):
    prompt: str
    user_id: int
    generation_params: dict | None = None

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu": "available"} # 简化

async def report_to_master(task_id: str, status: str, exit_code: int, log_tail: str = ""):
    payload = {
        "task_id": task_id,
        "status": status,
        "exit_code": exit_code,
        "message": log_tail,
    }
    url = MASTER_URL.rstrip("/") + "/task/report"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Report to master failed: {resp.status} {text}")
    except Exception as e:
        logger.error(f"Report to master exception: {e}")


async def report_eval_to_master(eval_id: str, status: str, exit_code: int, log_tail: str = ""):
    payload = {
        "eval_id": eval_id,
        "status": status,
        "exit_code": exit_code,
        "message": log_tail,
    }
    url = MASTER_URL.rstrip("/") + "/eval/report"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Report eval to master failed: {resp.status} {text}")
    except Exception as e:
        logger.error(f"Report eval to master exception: {e}")


async def report_export_to_master(export_id: str, status: str, exit_code: int, log_tail: str = ""):
    payload = {
        "export_id": export_id,
        "status": status,
        "exit_code": exit_code,
        "message": log_tail,
    }
    url = MASTER_URL.rstrip("/") + "/export/report"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Report export to master failed: {resp.status} {text}")
    except Exception as e:
        logger.error(f"Report export to master exception: {e}")

async def track_container(task_id: str, container_id: str):
    try:
        container = client.containers.get(container_id)
        # 使用线程等待容器退出，避免阻塞事件循环
        wait_result = await asyncio.to_thread(container.wait)
        exit_code = wait_result.get("StatusCode", -1)
        # 取末尾日志用于上报
        try:
            logs = container.logs(tail=50).decode("utf-8", errors="ignore")
        except Exception:
            logs = ""
        status = "success" if exit_code == 0 else "failed"
        # 识别 OOM 等错误，增强提示
        if exit_code == 137:
            logs = "可能显存不足(OOM)\n" + logs
        await report_to_master(task_id, status, exit_code, logs)
    except Exception as e:
        logger.error(f"track_container error for {task_id}: {e}")
    finally:
        # optional: delay before cleanup (cleanup loop also handles)
        await asyncio.sleep(5)


async def track_eval_container(eval_id: str, container_id: str):
    try:
        container = client.containers.get(container_id)
        wait_result = await asyncio.to_thread(container.wait)
        exit_code = wait_result.get("StatusCode", -1)
        try:
            logs = container.logs(tail=50).decode("utf-8", errors="ignore")
        except Exception:
            logs = ""
        status = "success" if exit_code == 0 else "failed"
        if exit_code == 137:
            logs = "可能显存不足(OOM)\n" + logs
        await report_eval_to_master(eval_id, status, exit_code, logs)
    except Exception as e:
        logger.error(f"track_eval_container error for {eval_id}: {e}")
    finally:
        await asyncio.sleep(5)


async def track_export_container(export_id: str, container_id: str):
    try:
        container = client.containers.get(container_id)
        wait_result = await asyncio.to_thread(container.wait)
        exit_code = wait_result.get("StatusCode", -1)
        try:
            logs = container.logs(tail=50).decode("utf-8", errors="ignore")
        except Exception:
            logs = ""
        status = "success" if exit_code == 0 else "failed"
        if exit_code == 137:
            logs = "可能显存不足(OOM)\n" + logs
        await report_export_to_master(export_id, status, exit_code, logs)
    except Exception as e:
        logger.error(f"track_export_container error for {export_id}: {e}")
    finally:
        await asyncio.sleep(5)


async def start_inference_container(
    deploy_id: str,
    user_id: int,
    config_yaml: str | None = None,
    model_path: str | None = None,
    api_port: int | None = None,
    api_host: str | None = None,
    api_key: str | None = None,
    api_model_name: str | None = None,
    api_verbose: str | None = None,
    max_concurrent: int | None = None,
    fastapi_root_path: str | None = None,
):
    volumes = {
        os.path.join(HOST_BASE_PATH, 'data'): {'bind': '/app/data', 'mode': 'ro'},
        os.path.join(HOST_BASE_PATH, 'output'): {'bind': '/app/output', 'mode': 'rw'},
        os.path.join(HOST_BASE_PATH, 'temp'): {'bind': '/app/temp', 'mode': 'ro'},
        HF_CACHE_PATH: {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
    }
    port_to_use = api_port or 8000
    api_host = api_host or "0.0.0.0"
    envs = ["HF_ENDPOINT=https://hf-mirror.com"]
    if api_key:
        envs.append(f"API_KEY={api_key}")
    if api_model_name:
        envs.append(f"API_MODEL_NAME={api_model_name}")
    if api_verbose:
        envs.append(f"API_VERBOSE={api_verbose}")
    if max_concurrent:
        envs.append(f"MAX_CONCURRENT={max_concurrent}")
    if fastapi_root_path:
        envs.append(f"FASTAPI_ROOT_PATH={fastapi_root_path}")

    config_path = None
    if config_yaml:
        # 清洗推理 YAML：移除 CLI 不接受的元信息字段
        try:
            loaded = yaml.safe_load(config_yaml) or {}
            for k in [
                "api_port",
                "api_host",
                "api_key",
                "api_model_name",
                "api_verbose",
                "max_concurrent",
                "fastapi_root_path",
            ]:
                loaded.pop(k, None)
            config_yaml = yaml.safe_dump(loaded, sort_keys=False, allow_unicode=False)
        except Exception as e:
            logger.error(f"sanitize infer yaml failed: {e}")
        # 确保 temp 目录存在（否则会出现 No such file or directory）
        temp_dir = os.path.join(HOST_BASE_PATH, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        config_path = os.path.join(temp_dir, f"{deploy_id}.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_yaml)
        cmd = f"llamafactory-cli api /app/temp/{deploy_id}.yaml"
    else:
        # 兼容旧逻辑：根据 model_path 构建命令
        if not model_path:
            raise HTTPException(status_code=400, detail="model_path required when config_yaml missing")
        resolved_model_path = model_path
        if model_path.startswith("/app/"):
            resolved_model_path = os.path.join(HOST_BASE_PATH, model_path[len("/app/"):].lstrip("/"))
        adapter_mode = os.path.isfile(os.path.join(resolved_model_path, "adapter_config.json"))
        base_model = None
        if adapter_mode:
            readme_path = os.path.join(resolved_model_path, "README.md")
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.lower().startswith("base_model:"):
                                base_model = line.split(":", 1)[1].strip()
                                break
                except Exception:
                    pass
            if not base_model:
                base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        cmd = (
            f"llamafactory-cli api --model_name_or_path {base_model} --adapter_name_or_path {model_path}"
            if adapter_mode else
            f"llamafactory-cli api --model_name_or_path {model_path}"
        )

    container_name = f"llm-infer-{user_id}-{deploy_id}"
    try:
        existing = client.containers.get(container_name)
        logger.info(f"found existing inference container {container_name}, removing before restart")
        existing.remove(force=True)
    except docker.errors.NotFound:
        pass

    logger.info(
        f"starting inference container for user {user_id} deploy {deploy_id} with cmd='{cmd}'"
    )
    container = client.containers.run(
        image="lf-with-optimum:v1.0",
        command=cmd,
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        volumes=volumes,
        detach=True,
        shm_size='2g',
        environment=envs,
        name=container_name,
        ports={f"{port_to_use}/tcp": None},
    )

    # 推理容器也落盘日志，便于启动异常时排查
    asyncio.create_task(collect_logs(f"inference-{deploy_id}", container.id))

    async def wait_port(max_wait: int = 60, inner_port: int = 8000):
        """等待容器端口映射并可 TCP 连接。"""
        deadline = datetime.datetime.utcnow() + datetime.timedelta(seconds=max_wait)
        host_port_val = 0
        while datetime.datetime.utcnow() < deadline:
            try:
                container.reload()
                status = container.status
                if status not in ["running", "created"]:
                    logger.error(f"inference container exited early with status={status}")
                    return 0
                port_info = container.attrs.get("NetworkSettings", {}).get("Ports", {}).get(f"{inner_port}/tcp", [])
                if port_info:
                    host_port_val = int(port_info[0].get("HostPort", 0))
                if host_port_val:
                    try:
                        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", host_port_val), timeout=2)
                        writer.close()
                        await writer.wait_closed()
                        return host_port_val
                    except Exception:
                        await asyncio.sleep(2)
                        continue
            except Exception:
                await asyncio.sleep(2)
                continue
            await asyncio.sleep(1)
        return host_port_val

    async def wait_http_ready(host_port_val: int, max_wait: int = 180):
        """端口有了以后，再检测 HTTP 接口是否可用，避免尚未加载权重就进入聊天。"""
        if not host_port_val:
            return False
        deadline = datetime.datetime.utcnow() + datetime.timedelta(seconds=max_wait)
        url = f"http://127.0.0.1:{host_port_val}/v1/chat/completions"
        payload = {
            "model": "local-infer",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        while datetime.datetime.utcnow() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=5) as resp:
                        if resp.status == 200:
                            return True
                        # 服务未就绪时通常 404/500，等待重试
            except Exception:
                await asyncio.sleep(2)
                continue
            await asyncio.sleep(2)
        return False

    host_port = await wait_port(inner_port=port_to_use)
    if not host_port:
        logger.error("inference container started but no host port mapped or service not ready")
        raise HTTPException(status_code=500, detail="推理容器启动失败或端口未就绪")

    http_ready = await wait_http_ready(host_port)
    if not http_ready:
        logger.error("inference container port open but HTTP not ready in time")
        raise HTTPException(status_code=500, detail="推理服务未就绪，请稍后重试")

    logger.info(f"inference container ready on host port {host_port} (HTTP ready)")
    return container, host_port


async def stop_inference_for_user(user_id: int):
    async with inference_lock:
        info = inference_sessions.get(user_id)
        if not info:
            return
        container_id = info.get("container_id")
        try:
            container = client.containers.get(container_id)
            container.stop()
            container.remove()
        except Exception as e:
            logger.error(f"stop inference container error: {e}")
        inference_sessions.pop(user_id, None)


async def collect_logs(task_id: str, container_id: str):
    """将容器日志持续写入本地文件，便于任务结束后查看。"""
    log_path = os.path.join(LOG_BASE_PATH, f"{task_id}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _consume():
        try:
            container = client.containers.get(container_id)
            with open(log_path, "ab") as f:
                for chunk in container.logs(stream=True, follow=True):
                    f.write(chunk)
                    f.flush()
        except Exception as e:
            logger.error(f"collect_logs error for {task_id}: {e}")

    await asyncio.to_thread(_consume)


async def inference_reaper():
    """超过 5 分钟无请求自动卸载推理容器。"""
    while True:
        now = datetime.datetime.utcnow()
        to_stop = []
        async with inference_lock:
            for user_id, info in list(inference_sessions.items()):
                last_used = info.get("last_used", now)
                if (now - last_used).total_seconds() > 300:
                    to_stop.append(user_id)
        for uid in to_stop:
            try:
                await stop_inference_for_user(uid)
            except Exception as e:
                logger.error(f"auto stop inference for user {uid} failed: {e}")
        await asyncio.sleep(30)

@app.post("/task/execute")
async def execute_task(task: TrainTask):
    try:
        logger.info(f"Received task: {task.task_id}")

        # 1. 准备目录
        task_temp_dir = os.path.join(HOST_BASE_PATH, "temp")
        os.makedirs(task_temp_dir, exist_ok=True)
        
        # 2. 写入配置文件
        config_path = os.path.join(task_temp_dir, f"{task.task_id}.yaml")
        with open(config_path, "w") as f:
            f.write(task.config_yaml)
        
        logger.info(f"Config written to {config_path}")

        # 3. 启动容器
        volumes = {
            os.path.join(HOST_BASE_PATH, 'data'): {'bind': '/app/data', 'mode': 'ro'},
            os.path.join(HOST_BASE_PATH, 'output'): {'bind': '/app/output', 'mode': 'rw'},
            os.path.join(HOST_BASE_PATH, 'temp'): {'bind': '/app/temp', 'mode': 'ro'},
            HF_CACHE_PATH: {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
        }

        container = client.containers.run(
            image="lf-with-optimum:v1.0",
            command=f"llamafactory-cli train /app/temp/{task.task_id}.yaml",
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            volumes=volumes,
            detach=True,
            shm_size='2g',
            environment=["HF_ENDPOINT=https://hf-mirror.com"],
            name=f"llm-worker-{task.task_id}"
        )
        
        logger.info(f"Container started: {container.id}")
        # 启动后台跟踪任务状态
        asyncio.create_task(track_container(task.task_id, container.id))
        # 启动后台日志落盘
        asyncio.create_task(collect_logs(task.task_id, container.id))
        return {"status": "started", "container_id": container.id}

    except Exception as e:
        logger.error(f"Failed to start task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eval/execute")
async def execute_eval(task: EvalTask):
    try:
        logger.info(f"Received eval: {task.eval_id}")

        task_temp_dir = os.path.join(HOST_BASE_PATH, "temp")
        os.makedirs(task_temp_dir, exist_ok=True)

        config_path = os.path.join(task_temp_dir, f"{task.eval_id}.yaml")
        with open(config_path, "w") as f:
            f.write(task.config_yaml)

        volumes = {
            os.path.join(HOST_BASE_PATH, 'data'): {'bind': '/app/data', 'mode': 'ro'},
            os.path.join(HOST_BASE_PATH, 'output'): {'bind': '/app/output', 'mode': 'rw'},
            os.path.join(HOST_BASE_PATH, 'temp'): {'bind': '/app/temp', 'mode': 'ro'},
            HF_CACHE_PATH: {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
        }

        container = client.containers.run(
            image="lf-with-optimum:v1.0",
            command=f"llamafactory-cli train /app/temp/{task.eval_id}.yaml",
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            volumes=volumes,
            detach=True,
            shm_size='2g',
            environment=["HF_ENDPOINT=https://hf-mirror.com"],
            name=f"llm-eval-{task.eval_id}"
        )

        logger.info(f"Eval container started: {container.id}")
        asyncio.create_task(track_eval_container(task.eval_id, container.id))
        asyncio.create_task(collect_logs(task.eval_id, container.id))
        return {"status": "started", "container_id": container.id}

    except Exception as e:
        logger.error(f"Failed to start eval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/execute")
async def execute_export(task: ExportTask):
    try:
        logger.info(f"Received export: {task.export_id}")

        task_temp_dir = os.path.join(HOST_BASE_PATH, "temp")
        os.makedirs(task_temp_dir, exist_ok=True)

        config_path = os.path.join(task_temp_dir, f"{task.export_id}.yaml")
        with open(config_path, "w") as f:
            f.write(task.config_yaml)

        volumes = {
            os.path.join(HOST_BASE_PATH, 'data'): {'bind': '/app/data', 'mode': 'ro'},
            os.path.join(HOST_BASE_PATH, 'output'): {'bind': '/app/output', 'mode': 'rw'},
            os.path.join(HOST_BASE_PATH, 'temp'): {'bind': '/app/temp', 'mode': 'ro'},
            HF_CACHE_PATH: {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
        }

        container = client.containers.run(
            image="lf-with-optimum:v1.0",
            command=f"llamafactory-cli export /app/temp/{task.export_id}.yaml",
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            volumes=volumes,
            detach=True,
            shm_size='2g',
            environment=["HF_ENDPOINT=https://hf-mirror.com"],
            name=f"llm-export-{task.export_id}"
        )

        logger.info(f"Export container started: {container.id}")
        asyncio.create_task(track_export_container(task.export_id, container.id))
        asyncio.create_task(collect_logs(task.export_id, container.id))
        return {"status": "started", "container_id": container.id}

    except Exception as e:
        logger.error(f"Failed to start export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/execute")
async def inference_execute(req: InferenceStart):
    # 同一用户只允许单实例
    async with inference_lock:
        if req.user_id in inference_sessions:
            info = inference_sessions[req.user_id]
            info["last_used"] = datetime.datetime.utcnow()
            return {"status": "exists", "port": info["port"], "container_id": info.get("container_id")}
    try:
        deploy_id = req.deploy_id or req.task_id
        container, host_port = await start_inference_container(
            deploy_id,
            req.user_id,
            req.config_yaml,
            req.model_path,
            req.api_port,
            req.api_host,
            req.api_key,
            req.api_model_name,
            req.api_verbose,
            req.max_concurrent,
            req.fastapi_root_path,
        )
        async with inference_lock:
            inference_sessions[req.user_id] = {
                "container_id": container.id,
                "port": host_port,
                "last_used": datetime.datetime.utcnow(),
            }
        return {"status": "started", "port": host_port, "container_id": container.id}
    except Exception as e:
        logger.error(f"inference_execute failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 兼容旧路径
@app.post("/inference/start")
async def inference_start(req: InferenceStart):
    return await inference_execute(req)


@app.post("/inference/stop")
async def inference_stop(req: InferenceStop):
    await stop_inference_for_user(req.user_id)
    return {"status": "stopped"}


@app.post("/inference/chat")
async def inference_chat(req: InferenceChat):
    async with inference_lock:
        info = inference_sessions.get(req.user_id)
    if not info:
        raise HTTPException(status_code=404, detail="No running inference")
    port = info.get("port")
    if not port:
        raise HTTPException(status_code=500, detail="Inference port not ready")
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "local-infer",
        "messages": [{"role": "user", "content": req.prompt}],
    }
    gen_params = req.generation_params or {}
    payload.update(gen_params)
    payload.setdefault("max_tokens", 256)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=20) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise HTTPException(status_code=resp.status, detail=data)
                async with inference_lock:
                    info = inference_sessions.get(req.user_id)
                    if info:
                        info["last_used"] = datetime.datetime.utcnow()
                return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"inference_chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/status/{task_id}")
def task_status(task_id: str):
    container_name = f"llm-worker-{task_id}"
    try:
        container = client.containers.get(container_name)
        container.reload()
        status = container.status
        exit_code = container.attrs.get("State", {}).get("ExitCode")
        return {"status": status, "exit_code": exit_code}
    except docker.errors.NotFound:
        return {"status": "not_found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def query_gpu_stats():
    """调用 nvidia-smi 获取 GPU 利用率与显存，失败则返回零值。"""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, encoding="utf-8").strip()
        first_line = out.splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        used, total, util, temp = [int(x) for x in parts[:4]]
        return used, total, util, temp
    except Exception as e:
        logger.error(f"query_gpu_stats failed: {e}")
        return 0, 0, 0, 0


def query_sys_stats():
    """查询 CPU 与内存占用，未安装 psutil 时降级为 0。"""
    try:
        if not psutil:
            return 0.0, 0.0
        cpu_percent = psutil.cpu_percent(interval=0.05)
        mem_percent = psutil.virtual_memory().percent
        return float(cpu_percent), float(mem_percent)
    except Exception as e:
        logger.error(f"query_sys_stats failed: {e}")
        return 0.0, 0.0


async def heartbeat_loop():
    """周期性上报心跳和 GPU 状态。"""
    while True:
        used, total, util, temp = query_gpu_stats()
        cpu_percent, mem_percent = query_sys_stats()
        payload = {
            "url": ADVERTISE_URL,
            "gpu_memory_used": used,
            "gpu_memory_total": total,
            "gpu_utilization": util,
            "gpu_temperature": temp,
            "cpu_percent": cpu_percent,
            "mem_percent": mem_percent,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(MASTER_URL.rstrip("/") + "/worker/heartbeat", json=payload, timeout=8) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Heartbeat failed: {resp.status} {text}")
        except Exception as e:
            logger.error(f"Heartbeat exception: {e}")
        await asyncio.sleep(10)


async def register_self():
    """启动时向 Master 注册自身 URL。"""
    payload = {"advertise_url": ADVERTISE_URL, "url": WORKER_URL}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(MASTER_URL.rstrip("/") + "/worker/register", json=payload, timeout=8) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Register failed: {resp.status} {text}")
    except Exception as e:
        logger.error(f"Register exception: {e}")

@app.websocket("/task/{task_id}/log")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        log_path = os.path.join(LOG_BASE_PATH, f"{task_id}.log")

        def tail_lines(path, limit=200):
            try:
                with open(path, "rb") as f:
                    lines = f.readlines()[-limit:]
                return [l.decode("utf-8", errors="ignore") for l in lines]
            except FileNotFoundError:
                return []

        # 先发历史尾部日志，避免重复发送过大内容
        for line in tail_lines(log_path):
            await websocket.send_text(line.rstrip("\n"))

        # 持续跟随文件追加的新内容
        last_size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        while True:
            try:
                if not os.path.exists(log_path):
                    await asyncio.sleep(1)
                    continue
                current_size = os.path.getsize(log_path)
                if current_size > last_size:
                    with open(log_path, "rb") as f:
                        f.seek(last_size)
                        chunk = f.read()
                        if chunk:
                            await websocket.send_text(chunk.decode("utf-8", errors="ignore"))
                    last_size = current_size
                await asyncio.sleep(0.5)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket log stream error: {e}")
                await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"WebSocket closed for {task_id}")
        try:
            await websocket.close()
        except:
            pass


@app.get("/task/{task_id}/log/download")
async def download_log(task_id: str):
    """提供日志文件下载，覆盖训练/评估/导出/推理任务。"""
    log_path = os.path.join(LOG_BASE_PATH, f"{task_id}.log")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="日志不存在")
    return FileResponse(log_path, filename=f"{task_id}.log", media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
