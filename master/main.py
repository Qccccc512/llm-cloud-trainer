from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Header, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from typing import List, Optional
import uuid
import datetime
import asyncio
import aiohttp
import os
import shutil
import tempfile
import hashlib
import secrets
import json
import re
import yaml
import subprocess
import sys
from sqlalchemy import or_

# 导入我们定义的模块
from master.database import create_db_and_tables, get_session, init_db_data, engine, DATA_DIR, sync_dataset_info_json
from master.models import TaskStatus, Task, WorkerStatus, Worker, User, EvalStatus, EvalJob, InferenceSession, InferenceStatus, ModelRegistry, ExportJob, ExportStatus, ResourceMetric, Dataset
from master.utils import generate_config, generate_eval_config
from master.utils import generate_export_config, generate_infer_config
from master.scheduler import scheduler_loop
from master.utils import resolve_model_alias

app = FastAPI(title="LLM Cloud Trainer Master")


def load_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_ipinfo(ipinfo_path: str) -> dict:
    """支持 yaml 或 key=value 文本。"""
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


# 读取配置（环境变量优先，其次配置文件，最后默认值）
MASTER_CONFIG_PATH = os.environ.get("MASTER_CONFIG", os.path.join(os.path.dirname(__file__), "config", "config.yaml"))
CONF = load_config(MASTER_CONFIG_PATH)
IPINFO_PATH = os.environ.get("IPINFO_PATH", os.path.join(os.path.dirname(__file__), "master-worker-ipinfo"))
IPINFO = load_ipinfo(IPINFO_PATH)
BASE_PATH = os.environ.get("CLOUD_LLM_ROOT", CONF.get("cloud_llm_root", "/home/ubuntu/CloudComputing/cloud-llm"))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", CONF.get("project_root", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))))
INVITE_CODE = os.environ.get("INVITE_CODE", CONF.get("invite_code", "llamaCloud"))
MASTER_PRIVATE_URL = os.environ.get(
    "MASTER_PRIVATE_URL",
    CONF.get("master_private_url") or IPINFO.get("master_private_url") or "http://localhost:8000",
)
MASTER_PUBLIC_URL = os.environ.get(
    "MASTER_PUBLIC_URL",
    CONF.get("master_public_url") or IPINFO.get("master_public_url") or MASTER_PRIVATE_URL,
)


def hash_password(plain: str) -> str:
    """使用随机 salt 的 SHA256 存储，格式 salt$hash。"""
    salt = secrets.token_hex(8)
    digest = hashlib.sha256((salt + plain).encode("utf-8")).hexdigest()
    return f"{salt}${digest}"


def verify_password(plain: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    calc = hashlib.sha256((salt + plain).encode("utf-8")).hexdigest()
    return secrets.compare_digest(calc, digest)


def require_user(authorization: str = Header(default=None), session: Session = Depends(get_session)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1].strip()
    user = session.exec(select(User).where(User.api_key == token)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


def get_user_by_token(token: str) -> Optional[User]:
    if not token:
        return None
    with Session(engine) as session:
        return session.exec(select(User).where(User.api_key == token)).first()


def _sanitize_dataset_name(name: str):
    if not name or not re.match(r"^[A-Za-z0-9._-]+$", name):
        raise HTTPException(status_code=400, detail="数据集名称仅支持字母、数字、.-_ 组合")


def _dataset_host_path(rel_path: str) -> str:
    return os.path.join(DATA_DIR, rel_path)

# 前端静态资源
app.mount("/static", StaticFiles(directory="master/static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("master/static/index.html")


# 初始化 DB
@app.on_event("startup")
async def on_startup():
    create_db_and_tables()
    init_db_data()
    # 以后这里会启动 scheduler
    asyncio.create_task(scheduler_loop())

# --- API 使用的 Pydantic 模型 ---
from pydantic import BaseModel, Field

class TaskCreate(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset: str = "identity"
    epochs: float = 3.0
    batch_size: int = 2
    learning_rate: float = 1e-4
    # 可选
    template: str = "qwen"
    config_overrides: dict = Field(default_factory=dict)

class WorkerRegister(BaseModel):
    url: Optional[str] = None  # 兼容旧字段，代表内网/优先访问地址
    advertise_url: Optional[str] = None  # 建议使用，内网优先

class TaskReport(BaseModel):
    task_id: str
    status: TaskStatus
    exit_code: Optional[int] = None
    message: Optional[str] = None

class WorkerHeartbeat(BaseModel):
    url: Optional[str] = None  # 内网/优先访问地址
    cpu_percent: float = 0.0
    mem_percent: float = 0.0
    gpu_memory_used: int = 0
    gpu_memory_total: int = 0
    gpu_utilization: int = 0
    gpu_temperature: int = 0
    status: Optional[WorkerStatus] = None

class ModelCreate(BaseModel):
    name: str
    path: Optional[str] = None
    description: Optional[str] = None
    hf_path: Optional[str] = None
    ms_path: Optional[str] = None

class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str
    invite_code: str

class LoginRequest(BaseModel):
    username: str
    password: str

class InferenceDeploy(BaseModel):
    task_id: str
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    template: str = "qwen"
    adapter_path: Optional[str] = None
    infer_backend: str = "huggingface"
    api_port: Optional[int] = None
    api_host: Optional[str] = None
    config_overrides: dict = Field(default_factory=dict)

class InferenceStop(BaseModel):
    session_id: Optional[int] = None

class InferenceChat(BaseModel):
    prompt: str
    session_id: Optional[int] = None
    generation_params: Optional[dict] = Field(default=None)


class EvalCreate(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_path: Optional[str] = None
    template: str = "qwen"
    dataset: str = "identity"
    seed: int = 42
    batch_size: int = 4
    max_samples: int = 200
    config_overrides: dict = Field(default_factory=dict)


class EvalReport(BaseModel):
    eval_id: str
    status: EvalStatus
    exit_code: Optional[int] = None
    message: Optional[str] = None


class ExportCreate(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: Optional[str] = None
    template: str = "qwen"
    quantization_bit: Optional[int] = None
    quantization_dataset: Optional[str] = None
    config_overrides: dict = Field(default_factory=dict)


class ExportReport(BaseModel):
    export_id: str
    status: ExportStatus
    exit_code: Optional[int] = None
    message: Optional[str] = None


class DatasetRegister(BaseModel):
    name: str
    visibility: str = "private"  # 可选值 public/private
    source: str = "local"
    path: Optional[str] = None
    info: dict = Field(default_factory=dict)
    remark: Optional[str] = None


class DatasetRemarkUpdate(BaseModel):
    remark: Optional[str] = None


class DatasetPreviewResponse(BaseModel):
    dataset_id: int
    name: str
    file_name: Optional[str] = None
    kind: str
    preview: list


class DatasetCleanRequest(BaseModel):
    new_name: str
    remark: Optional[str] = None


class DatasetGenerateRequest(BaseModel):
    new_name: str
    api_key: str
    base_url: str
    model: str = "deepseek-chat"
    lang: str = "zh-CN"
    profile: str = "mixed"
    style: str = ""
    chunk_chars: int = 2500
    overlap: int = 200
    pairs_per_chunk: int = 3
    max_examples: int = 200
    max_output_tokens: int = 1200
    remark: Optional[str] = None


class DatasetAugmentRequest(BaseModel):
    new_name: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    few_shot_num: int = 10
    similarity_threshold: float = 0.7
    generate_num: int = 10
    remark: Optional[str] = None

# --- API 路由 ---

@app.post("/auth/register")
def register_user(body: RegisterRequest, session: Session = Depends(get_session)):
    if body.invite_code != INVITE_CODE:
        raise HTTPException(status_code=400, detail="邀请码错误")
    if not (6 <= len(body.password) <= 20):
        raise HTTPException(status_code=400, detail="密码长度需 6-20 位")
    if body.password != body.confirm_password:
        raise HTTPException(status_code=400, detail="两次密码不一致")

    existing = session.exec(select(User).where(User.username == body.username)).first()
    if existing:
        raise HTTPException(status_code=400, detail="用户名已存在")

    api_key = uuid.uuid4().hex
    user = User(username=body.username, api_key=api_key, password_hash=hash_password(body.password))
    session.add(user)
    session.commit()
    session.refresh(user)
    return {"token": api_key, "username": user.username, "user_id": user.id}


@app.post("/auth/login")
def login_user(body: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.username == body.username)).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    user.api_key = uuid.uuid4().hex
    session.add(user)
    session.commit()
    session.refresh(user)
    return {"token": user.api_key, "username": user.username, "user_id": user.id}


@app.get("/me")
def current_user_profile(user: User = Depends(require_user)):
    return {"username": user.username, "user_id": user.id}

@app.post("/submit", response_model=Task)
def submit_task(task_params: TaskCreate, session: Session = Depends(get_session), user: User = Depends(require_user)):
    """
    用户提交训练任务
    """
    new_id = f"task_{uuid.uuid4().hex[:8]}"
    
    payload = task_params.model_dump()
    payload["model_name"] = resolve_model_alias(task_params.model_name)
    ensure_train_model_allowed(payload["model_name"], session, user)
    config_json_str = json.dumps(payload)
    
    task = Task(
        id=new_id,
        status=TaskStatus.PENDING,
        config_json=config_json_str,
        created_at=datetime.datetime.utcnow(),
        user_id=user.id
    )
    
    session.add(task)
    session.commit()
    session.refresh(task)
    return task

@app.get("/tasks", response_model=List[Task])
def list_tasks(session: Session = Depends(get_session), user: User = Depends(require_user)):
    """查看当前用户任务状态"""
    tasks = session.exec(
        select(Task).where(Task.user_id == user.id).order_by(Task.created_at.desc())
    ).all()
    return tasks

@app.post("/worker/register")
def register_worker(worker_data: WorkerRegister, session: Session = Depends(get_session)):
    """Worker 启动时主动注册，内网地址优先用于调度。"""
    advertise_url = _normalize_url(worker_data.advertise_url or worker_data.url)
    urls = [_normalize_url(u) for u in {advertise_url, worker_data.url} if u]
    if not urls:
        raise HTTPException(status_code=400, detail="worker url missing")

    canonical_url = advertise_url or urls[0]
    existing = _dedupe_workers(session, urls)

    if existing:
        existing.status = WorkerStatus.IDLE
        existing.last_heartbeat = datetime.datetime.utcnow()
        existing.url = canonical_url
        session.add(existing)
    else:
        session.add(Worker(url=canonical_url, status=WorkerStatus.IDLE))

    session.commit()
    return {"status": "registered"}

@app.get("/workers")
def list_workers(session: Session = Depends(get_session), user: User = Depends(require_user)):
    workers = session.exec(select(Worker)).all()
    return [
        {
            "id": w.id,
            "status": w.status,
            "last_heartbeat": w.last_heartbeat,
            "url": w.url,
            "cpu_percent": w.cpu_percent,
            "mem_percent": w.mem_percent,
            "gpu_memory_used": w.gpu_memory_used,
            "gpu_memory_total": w.gpu_memory_total,
            "gpu_utilization": w.gpu_utilization,
            "gpu_temperature": w.gpu_temperature,
        }
        for w in workers
    ]


def _normalize_url(u: Optional[str]) -> str:
    if not u:
        return ""
    # 统一去掉末尾斜杠，保留协议/主机/端口
    return u.strip().rstrip("/")


def _dedupe_workers(session: Session, urls: List[str]) -> Optional[Worker]:
    """按规范化 URL 去重，返回保留的 Worker，其余删除。"""
    norms = {_normalize_url(u) for u in urls if u}
    if not norms:
        return None
    q = select(Worker).where(Worker.url.in_(list(norms)))
    rows = session.exec(q).all()
    if not rows:
        return None
    # 按 id 升序保留第一条，删除其它重复
    rows_sorted = sorted(rows, key=lambda w: (w.id or 0))
    keeper = rows_sorted[0]
    for extra in rows_sorted[1:]:
        session.delete(extra)
    return keeper


def pick_available_worker(session: Session) -> Optional[Worker]:
    worker = session.exec(select(Worker).where(Worker.status != WorkerStatus.OFFLINE)).first()
    return worker


def parse_output_id(path: Optional[str]) -> Optional[str]:
    """从 /app/output/<id>/... 路径中解析出任务或导出 ID"""
    if not path:
        return None
    marker = "/app/output/"
    if marker not in path:
        return None
    remainder = path.split(marker, 1)[1]
    return remainder.split("/", 1)[0] or None


def is_builtin_model(model_name: Optional[str], session: Session) -> bool:
    """校验是否为预置模型（name/path/hf/ms 均视为合法别名）"""
    if not model_name:
        return False
    models = session.exec(select(ModelRegistry)).all()
    aliases = []
    for m in models:
        aliases.extend([m.name, m.path, m.hf_path, m.ms_path])
    return model_name in [a for a in aliases if a]


def ensure_train_model_allowed(model_name: str, session: Session, user: User):
    """训练仅允许预置模型或本人成功导出的产物"""
    if is_builtin_model(model_name, session):
        return
    export_id = parse_output_id(model_name)
    if export_id:
        export_job = session.get(ExportJob, export_id)
        if export_job and export_job.user_id == user.id and export_job.status == ExportStatus.SUCCESS:
            return
    raise HTTPException(status_code=400, detail="仅可选择预置模型或本人导出产物")


def ensure_infer_model_allowed(model_name: Optional[str], model_path: Optional[str], session: Session, user: User):
    """评测/部署仅允许预置模型、本人成功的训练或导出产物"""
    # 预置模型：无 /app/output 路径且匹配模型表
    if (not model_path or not model_path.startswith("/app/output/")) and is_builtin_model(model_name or model_path, session):
        return

    target_id = parse_output_id(model_path or model_name)
    if target_id:
        export_job = session.get(ExportJob, target_id)
        if export_job and export_job.user_id == user.id and export_job.status == ExportStatus.SUCCESS:
            return
        task = session.get(Task, target_id)
        if task and task.user_id == user.id and task.status == TaskStatus.SUCCESS:
            return

    raise HTTPException(status_code=400, detail="模型仅限预置/本人训练产物/本人导出产物")


def ensure_export_allowed(model_name: str, adapter_path: Optional[str], session: Session, user: User):
    """导出仅允许预置模型或本人产物，LoRA 需验证归属"""
    if adapter_path:
        task_id = parse_output_id(adapter_path)
        task = session.get(Task, task_id) if task_id else None
        if not (task and task.user_id == user.id and task.status == TaskStatus.SUCCESS):
            raise HTTPException(status_code=400, detail="仅可导出本人成功的训练任务")
        return

    if is_builtin_model(model_name, session):
        return

    export_id = parse_output_id(model_name)
    if export_id:
        export_job = session.get(ExportJob, export_id)
        if export_job and export_job.user_id == user.id and export_job.status == ExportStatus.SUCCESS:
            return

    raise HTTPException(status_code=400, detail="仅可导出预置模型或本人导出产物")


async def dispatch_eval_job(eval_job: EvalJob, session: Session):
    worker = pick_available_worker(session)
    if not worker:
        raise HTTPException(status_code=503, detail="无可用 Worker")

    config_yaml = generate_eval_config(eval_job.id, eval_job.config_json)
    payload = {"eval_id": eval_job.id, "config_yaml": config_yaml}

    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(f"{worker.url.rstrip('/')}/eval/execute", json=payload, timeout=10) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    eval_job.status = EvalStatus.RUNNING
    eval_job.worker_id = worker.id
    eval_job.updated_at = datetime.datetime.utcnow()
    worker.status = WorkerStatus.BUSY
    worker.last_heartbeat = datetime.datetime.utcnow()
    session.add(eval_job)
    session.add(worker)
    session.commit()
    session.refresh(eval_job)
    return eval_job


async def dispatch_export_job(export_job: ExportJob, session: Session):
    worker = pick_available_worker(session)
    if not worker:
        raise HTTPException(status_code=503, detail="无可用 Worker")

    config_yaml = generate_export_config(export_job.id, export_job.config_json)
    payload = {"export_id": export_job.id, "config_yaml": config_yaml}

    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(f"{worker.url.rstrip('/')}/export/execute", json=payload, timeout=10) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    export_job.status = ExportStatus.RUNNING
    export_job.worker_id = worker.id
    export_job.updated_at = datetime.datetime.utcnow()
    worker.status = WorkerStatus.BUSY
    worker.last_heartbeat = datetime.datetime.utcnow()
    session.add(export_job)
    session.add(worker)
    session.commit()
    session.refresh(export_job)
    return export_job


@app.post("/worker/heartbeat")
def worker_heartbeat(hb: WorkerHeartbeat, session: Session = Depends(get_session)):
    """Worker 上报心跳与 GPU 状态。"""
    advertise_url = _normalize_url(hb.url)
    urls = [_normalize_url(u) for u in {advertise_url} if u]

    worker = _dedupe_workers(session, urls)
    if not worker:
        worker = Worker(url=advertise_url or (urls[0] if urls else ""), status=WorkerStatus.IDLE)
        session.add(worker)

    worker.cpu_percent = hb.cpu_percent
    worker.mem_percent = hb.mem_percent
    worker.gpu_memory_used = hb.gpu_memory_used
    worker.gpu_memory_total = hb.gpu_memory_total or worker.gpu_memory_total
    worker.gpu_utilization = hb.gpu_utilization
    worker.gpu_temperature = hb.gpu_temperature
    worker.last_heartbeat = datetime.datetime.utcnow()
    if hb.status:
        worker.status = hb.status

    session.add(worker)
    session.flush()  # 中文注释：确保拿到 worker.id 再写资源日志

    metric = ResourceMetric(
        worker_id=worker.id,
        cpu_percent=hb.cpu_percent,
        mem_percent=hb.mem_percent,
        gpu_utilization=hb.gpu_utilization,
        gpu_memory_used=hb.gpu_memory_used,
        gpu_memory_total=hb.gpu_memory_total,
        gpu_temperature=hb.gpu_temperature,
    )
    session.add(metric)
    session.commit()
    return {"status": "ok"}


@app.post("/task/report")
def report_task_status(report: TaskReport, session: Session = Depends(get_session)):
    task = session.get(Task, report.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task.status = report.status
    task.updated_at = datetime.datetime.utcnow()

    if task.worker_id:
        worker = session.get(Worker, task.worker_id)
        if worker:
            worker.status = WorkerStatus.IDLE
            worker.last_heartbeat = datetime.datetime.utcnow()
            session.add(worker)

    session.add(task)
    session.commit()
    return {"status": "ok"}


@app.get("/task/{task_id}", response_model=Task)
def get_task(task_id: str, session: Session = Depends(get_session), user: User = Depends(require_user)):
    task = session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return task


@app.get("/download/{task_id}")
def download_artifact(task_id: str, background_tasks: BackgroundTasks, token: Optional[str] = None, authorization: str = Header(default=None), session: Session = Depends(get_session)):
    # 兼容查询参数 token 或 Header Bearer token
    user: Optional[User] = None
    if token:
        user = get_user_by_token(token)
    elif authorization and authorization.startswith("Bearer "):
        auth_token = authorization.split(" ", 1)[1].strip()
        user = get_user_by_token(auth_token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    task = session.get(Task, task_id)
    if not task or task.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    """打包并下载指定任务的输出目录。"""
    output_dir = os.path.join(BASE_PATH, "output", task_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Output not found")

    # 创建临时 zip 包
    tmp_base = tempfile.mktemp(prefix=f"{task_id}_", suffix="")
    zip_path = shutil.make_archive(tmp_base, "zip", root_dir=output_dir)

    # 响应结束后删除临时文件
    background_tasks.add_task(lambda p: os.path.exists(p) and os.remove(p), zip_path)

    return FileResponse(zip_path, media_type="application/zip", filename=f"{task_id}.zip")


@app.get("/datasets")
def list_datasets(session: Session = Depends(get_session), authorization: str = Header(default=None)):
    """未登录用户可浏览公共数据集，登录后可见自己的私有数据集。"""
    user: Optional[User] = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        user = get_user_by_token(token)

    stmt = select(Dataset).where(Dataset.visibility == "public")
    if user:
        stmt = select(Dataset).where(or_(Dataset.visibility == "public", Dataset.owner_id == user.id))
    datasets = session.exec(stmt.order_by(Dataset.created_at.desc())).all()
    return datasets


@app.post("/datasets/upload", response_model=Dataset)
async def upload_dataset(
    name: str = Form(...),
    remark: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    _sanitize_dataset_name(name)
    exists = session.exec(select(Dataset).where(Dataset.name == name, Dataset.owner_id == user.id)).first()
    if exists:
        raise HTTPException(status_code=400, detail="同名私有数据集已存在")

    dest_dir = os.path.join(DATA_DIR, f"user_{user.id}", name)
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(file.filename) or f"{name}.json"
    dest_path = os.path.join(dest_dir, filename)
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存文件失败: {e}")

    rel = os.path.relpath(dest_path, DATA_DIR)
    info = {"file_name": rel}
    ds = Dataset(
        name=name,
        display_name=name,
        visibility="private",
        source="upload",
        path=f"/app/data/{rel}",
        info=info,
        remark=remark,
        owner_id=user.id,
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow(),
    )
    session.add(ds)
    session.commit()
    session.refresh(ds)
    sync_dataset_info_json(session)
    return ds


@app.post("/datasets/register", response_model=Dataset)
def register_dataset(body: DatasetRegister, session: Session = Depends(get_session), user: User = Depends(require_user)):
    _sanitize_dataset_name(body.name)
    visibility = body.visibility
    if visibility == "public" and user.username != "admin":
        # 非 admin 注册公共数据集会自动降级为私有
        visibility = "private"
    owner_id = None if visibility == "public" else user.id

    exists = session.exec(select(Dataset).where(Dataset.name == body.name, Dataset.owner_id == owner_id)).first()
    if exists:
        raise HTTPException(status_code=400, detail="数据集已存在")

    info = body.info or {}
    path = body.path
    if not path and isinstance(info, dict) and info.get("file_name"):
        path = f"/app/data/{info['file_name']}"

    ds = Dataset(
        name=body.name,
        display_name=body.name,
        visibility=visibility,
        source=body.source,
        path=path,
        info=info,
        remark=body.remark,
        owner_id=owner_id,
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow(),
    )
    session.add(ds)
    session.commit()
    session.refresh(ds)
    sync_dataset_info_json(session)
    return ds


@app.patch("/datasets/{dataset_id}/remark", response_model=Dataset)
def update_dataset_remark(dataset_id: int, body: DatasetRemarkUpdate, session: Session = Depends(get_session), user: User = Depends(require_user)):
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if ds.visibility == "public" and user.username != "admin":
        raise HTTPException(status_code=403, detail="无权限修改公共数据集")
    if ds.visibility == "private" and ds.owner_id != user.id:
        raise HTTPException(status_code=403, detail="无权限修改他人私有数据集")

    ds.remark = body.remark
    ds.updated_at = datetime.datetime.utcnow()
    session.add(ds)
    session.commit()
    session.refresh(ds)
    return ds


@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, session: Session = Depends(get_session), user: User = Depends(require_user)):
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if ds.visibility == "public":
        raise HTTPException(status_code=400, detail="公共数据集不允许删除")
    if ds.owner_id != user.id:
        raise HTTPException(status_code=403, detail="无权限删除他人数据集")

    if ds.source == "upload":
        base_dir = os.path.join(DATA_DIR, f"user_{user.id}", ds.name)
        try:
            if os.path.commonpath([base_dir, DATA_DIR]) == DATA_DIR and os.path.isdir(base_dir):
                shutil.rmtree(base_dir, ignore_errors=True)
        except Exception:
            pass

    session.delete(ds)
    session.commit()
    sync_dataset_info_json(session)
    return {"status": "deleted"}


def _ensure_dataset_visible(ds: Dataset, user: Optional[User]):
    if ds.visibility == "public":
        return
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if ds.owner_id != user.id:
        raise HTTPException(status_code=403, detail="无权限访问他人私有数据集")


def _ensure_dataset_owned(ds: Dataset, user: User):
    if ds.visibility != "private" or ds.owner_id != user.id:
        raise HTTPException(status_code=403, detail="仅允许操作本人私有数据集")


def _resolve_dataset_file_host_path(ds: Dataset) -> (str, str):
    """返回 (host_path, rel_path)。仅支持落盘在 DATA_DIR 下的文件。"""
    rel = None
    if isinstance(ds.info, dict) and ds.info.get("file_name"):
        rel = str(ds.info.get("file_name"))
    elif ds.path and isinstance(ds.path, str) and ds.path.startswith("/app/data/"):
        rel = ds.path[len("/app/data/"):].lstrip("/")
    if not rel:
        raise HTTPException(status_code=400, detail="该数据集不是本地文件，暂不支持预览/下载/派生")

    host_path = _dataset_host_path(rel)
    try:
        if os.path.commonpath([os.path.abspath(host_path), os.path.abspath(DATA_DIR)]) != os.path.abspath(DATA_DIR):
            raise HTTPException(status_code=400, detail="非法数据集路径")
    except Exception:
        raise HTTPException(status_code=400, detail="非法数据集路径")

    if not os.path.isfile(host_path):
        raise HTTPException(status_code=404, detail="数据集文件不存在")

    return host_path, rel


def _create_derived_dataset(
    *,
    session: Session,
    user: User,
    new_name: str,
    output_host_path: str,
    remark: Optional[str] = None,
    source: str = "derived",
    extra_info: Optional[dict] = None,
) -> Dataset:
    _sanitize_dataset_name(new_name)
    exists = session.exec(select(Dataset).where(Dataset.name == new_name, Dataset.owner_id == user.id)).first()
    if exists:
        raise HTTPException(status_code=400, detail="同名私有数据集已存在")

    abs_out = os.path.abspath(output_host_path)
    abs_data_dir = os.path.abspath(DATA_DIR)
    if os.path.commonpath([abs_out, abs_data_dir]) != abs_data_dir:
        raise HTTPException(status_code=400, detail="输出文件必须位于 data 目录下")
    if not os.path.isfile(abs_out):
        raise HTTPException(status_code=500, detail="派生产物文件未生成")

    rel = os.path.relpath(abs_out, DATA_DIR)
    info = {"file_name": rel}
    if isinstance(extra_info, dict):
        info.update(extra_info)

    ds = Dataset(
        name=new_name,
        display_name=new_name,
        visibility="private",
        source=source,
        path=f"/app/data/{rel}",
        info=info,
        remark=remark,
        owner_id=user.id,
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow(),
    )
    session.add(ds)
    session.commit()
    session.refresh(ds)
    sync_dataset_info_json(session)
    return ds


def _get_user_from_auth_header(authorization: Optional[str]) -> Optional[User]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    return get_user_by_token(token)


@app.get("/datasets/{dataset_id}/download")
def download_dataset_file(
    dataset_id: int,
    token: Optional[str] = None,
    session: Session = Depends(get_session),
    authorization: str = Header(default=None),
):
    user = _get_user_from_auth_header(authorization)
    if not user and token:
        user = get_user_by_token(token)
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    _ensure_dataset_visible(ds, user)
    host_path, rel = _resolve_dataset_file_host_path(ds)
    filename = os.path.basename(rel) or f"{ds.name}.data"
    return FileResponse(host_path, filename=filename)


@app.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def preview_dataset_file(
    dataset_id: int,
    limit: int = 50,
    session: Session = Depends(get_session),
    authorization: str = Header(default=None),
):
    user = _get_user_from_auth_header(authorization)
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    _ensure_dataset_visible(ds, user)
    host_path, rel = _resolve_dataset_file_host_path(ds)
    limit = max(1, min(int(limit or 50), 200))
    ext = os.path.splitext(host_path)[1].lower()

    preview: list = []
    kind = "text"

    if ext == ".jsonl":
        kind = "jsonl"
        with open(host_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(limit):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    preview.append(json.loads(line))
                except Exception:
                    preview.append(line)
    elif ext == ".json":
        kind = "json"
        size = os.path.getsize(host_path)
        if size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件过大，暂不支持 JSON 预览（>5MB）")
        with open(host_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if isinstance(data, list):
            preview = data[:limit]
        elif isinstance(data, dict):
            # 返回前 N 个键值对
            for i, (k, v) in enumerate(data.items()):
                if i >= limit:
                    break
                preview.append({k: v})
        else:
            preview = [data]
    else:
        kind = "text"
        with open(host_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(limit):
                line = f.readline()
                if not line:
                    break
                preview.append(line.rstrip("\n"))

    return {
        "dataset_id": ds.id,
        "name": ds.name,
        "file_name": rel,
        "kind": kind,
        "preview": preview,
    }


@app.post("/datasets/{dataset_id}/clean", response_model=Dataset)
def clean_dataset(
    dataset_id: int,
    body: DatasetCleanRequest,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    _ensure_dataset_owned(ds, user)
    input_host_path, _ = _resolve_dataset_file_host_path(ds)

    out_dir = os.path.join(DATA_DIR, f"user_{user.id}", body.new_name)
    os.makedirs(out_dir, exist_ok=True)

    script_path = os.path.join(PROJECT_ROOT, "z_todo", "datawash", "wash.py")
    if not os.path.isfile(script_path):
        raise HTTPException(status_code=500, detail="未找到清洗脚本")

    cmd = [sys.executable, script_path, "--input", input_host_path, "--output", out_dir]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"执行清洗失败: {e}")

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise HTTPException(status_code=400, detail=f"清洗失败: {msg[-1200:]}")

    # 取出产物：优先 cleaned_*.txt
    candidates = []
    for fn in os.listdir(out_dir):
        if fn.lower().endswith(".txt") and fn.startswith("cleaned_"):
            candidates.append(os.path.join(out_dir, fn))
    if not candidates:
        # 兜底：取任意 txt
        for fn in os.listdir(out_dir):
            if fn.lower().endswith(".txt"):
                candidates.append(os.path.join(out_dir, fn))
    if not candidates:
        raise HTTPException(status_code=500, detail="清洗完成但未生成输出文件")

    output_file = sorted(candidates)[0]
    remark = body.remark or f"由 {ds.name} 清洗生成"
    return _create_derived_dataset(
        session=session,
        user=user,
        new_name=body.new_name,
        output_host_path=output_file,
        remark=remark,
        source="clean",
        extra_info={"derived_from": ds.name},
    )


@app.post("/datasets/{dataset_id}/generate", response_model=Dataset)
def generate_dataset_from_text(
    dataset_id: int,
    body: DatasetGenerateRequest,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    _ensure_dataset_owned(ds, user)
    input_host_path, _ = _resolve_dataset_file_host_path(ds)

    out_dir = os.path.join(DATA_DIR, f"user_{user.id}", body.new_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{body.new_name}.jsonl")

    script_path = os.path.join(PROJECT_ROOT, "z_todo", "datawash", "generate_alpaca_jsonl.py")
    if not os.path.isfile(script_path):
        raise HTTPException(status_code=500, detail="未找到生成脚本")

    cmd = [
        sys.executable,
        script_path,
        "--input",
        input_host_path,
        "--output",
        out_path,
        "--api-key",
        body.api_key,
        "--base-url",
        body.base_url,
        "--model",
        body.model,
        "--lang",
        body.lang,
        "--profile",
        body.profile,
        "--style",
        body.style or "",
        "--chunk-chars",
        str(int(body.chunk_chars)),
        "--overlap",
        str(int(body.overlap)),
        "--pairs-per-chunk",
        str(int(body.pairs_per_chunk)),
        "--max-examples",
        str(int(body.max_examples)),
        "--max-output-tokens",
        str(int(body.max_output_tokens)),
    ]

    # 避免 key 被子进程写入日志（脚本本身不打印 key）；主进程也不打印 cmd。
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"执行生成失败: {e}")

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise HTTPException(status_code=400, detail=f"生成失败: {msg[-1600:]}")

    remark = body.remark or f"由 {ds.name} 生成 Alpaca 数据集"
    return _create_derived_dataset(
        session=session,
        user=user,
        new_name=body.new_name,
        output_host_path=out_path,
        remark=remark,
        source="generate",
        extra_info={"derived_from": ds.name, "format": "alpaca", "file_type": "jsonl"},
    )


@app.post("/datasets/{dataset_id}/augment", response_model=Dataset)
def augment_alpaca_dataset(
    dataset_id: int,
    body: DatasetAugmentRequest,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    ds = session.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    _ensure_dataset_owned(ds, user)
    input_host_path, _ = _resolve_dataset_file_host_path(ds)

    ext = os.path.splitext(input_host_path)[1].lower()
    if ext not in {".json", ".jsonl"}:
        raise HTTPException(status_code=400, detail="增强仅支持 .json 或 .jsonl")

    out_dir = os.path.join(DATA_DIR, f"user_{user.id}", body.new_name)
    os.makedirs(out_dir, exist_ok=True)
    # 脚本输出 JSON 数组；我们统一再导出为 jsonl 以便训练/查看。
    tmp_input_json = os.path.join(out_dir, f"{body.new_name}.__input.json")
    tmp_output_json = os.path.join(out_dir, f"{body.new_name}.__output.json")
    out_path = os.path.join(out_dir, f"{body.new_name}.jsonl")

    # 若源是 jsonl，则先转为 JSON 数组喂给脚本；若源是 json，直接使用。
    script_input = input_host_path
    if ext == ".jsonl":
        items = []
        try:
            with open(input_host_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"读取 jsonl 失败: {e}")
        if not items:
            raise HTTPException(status_code=400, detail="jsonl 为空或无有效 JSON 行")
        try:
            with open(tmp_input_json, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"写入临时 JSON 失败: {e}")
        script_input = tmp_input_json

    script_path = os.path.join(PROJECT_ROOT, "z_todo", "task", "data_augmentation.py")
    if not os.path.isfile(script_path):
        raise HTTPException(status_code=500, detail="未找到增强脚本")

    cmd = [
        sys.executable,
        script_path,
        "--input",
        script_input,
        "--output",
        tmp_output_json,
        "--api_key",
        body.api_key,
        "--base_url",
        body.base_url,
        "--model",
        body.model,
        "--few_shot_num",
        str(int(body.few_shot_num)),
        "--similarity_threshold",
        str(float(body.similarity_threshold)),
        "--generate_num",
        str(int(body.generate_num)),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"执行增强失败: {e}")

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise HTTPException(status_code=400, detail=f"增强失败: {msg[-1600:]}")

    # 将脚本输出（JSON 数组）导出为 jsonl
    try:
        with open(tmp_output_json, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("增强输出不是 JSON 数组")
        with open(out_path, "w", encoding="utf-8") as out_f:
            for item in data:
                out_f.write(json.dumps(item, ensure_ascii=False))
                out_f.write("\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出 jsonl 失败: {e}")

    remark = body.remark or f"由 {ds.name} 增强生成"
    return _create_derived_dataset(
        session=session,
        user=user,
        new_name=body.new_name,
        output_host_path=out_path,
        remark=remark,
        source="augment",
        extra_info={"derived_from": ds.name, "format": "alpaca", "file_type": "jsonl"},
    )


@app.post("/eval/submit", response_model=EvalJob)
async def submit_eval_job(body: EvalCreate, session: Session = Depends(get_session), user: User = Depends(require_user)):
    new_id = f"eval_{uuid.uuid4().hex[:8]}"
    payload = body.model_dump()
    ensure_infer_model_allowed(payload.get("model_name"), payload.get("model_path"), session, user)
    if not payload.get("model_path"):
        payload["model_name"] = resolve_model_alias(body.model_name)
    config_json_str = json.dumps(payload)
    eval_job = EvalJob(
        id=new_id,
        status=EvalStatus.PENDING,
        config_json=config_json_str,
        created_at=datetime.datetime.utcnow(),
        user_id=user.id,
    )
    session.add(eval_job)
    session.commit()
    session.refresh(eval_job)
    # 直接分发评估任务
    await dispatch_eval_job(eval_job, session)
    return eval_job


@app.post("/export/submit", response_model=ExportJob)
async def submit_export_job(body: ExportCreate, session: Session = Depends(get_session), user: User = Depends(require_user)):
    new_id = f"export_{uuid.uuid4().hex[:8]}"
    payload = body.model_dump()
    payload["model_name"] = resolve_model_alias(body.model_name)
    ensure_export_allowed(payload["model_name"], payload.get("adapter_path"), session, user)
    # 若用户选择量化且传入适配器路径（LoRA），拦截并提示先合并
    if payload.get("quantization_bit") and payload.get("adapter_path"):
        raise HTTPException(status_code=400, detail="先合并导出再量化")
    config_json_str = json.dumps(payload)
    export_job = ExportJob(
        id=new_id,
        status=ExportStatus.PENDING,
        config_json=config_json_str,
        created_at=datetime.datetime.utcnow(),
        user_id=user.id,
    )
    session.add(export_job)
    session.commit()
    session.refresh(export_job)
    await dispatch_export_job(export_job, session)
    return export_job


@app.get("/evals", response_model=List[EvalJob])
def list_eval_jobs(session: Session = Depends(get_session), user: User = Depends(require_user)):
    jobs = session.exec(select(EvalJob).where(EvalJob.user_id == user.id).order_by(EvalJob.created_at.desc())).all()
    return jobs


@app.get("/exports", response_model=List[ExportJob])
def list_export_jobs(session: Session = Depends(get_session), user: User = Depends(require_user)):
    jobs = session.exec(select(ExportJob).where(ExportJob.user_id == user.id).order_by(ExportJob.created_at.desc())).all()
    return jobs


@app.get("/eval/{eval_id}", response_model=EvalJob)
def get_eval_job(eval_id: str, session: Session = Depends(get_session), user: User = Depends(require_user)):
    job = session.get(EvalJob, eval_id)
    if not job:
        raise HTTPException(status_code=404, detail="Eval not found")
    if job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return job


@app.get("/export/{export_id}", response_model=ExportJob)
def get_export_job(export_id: str, session: Session = Depends(get_session), user: User = Depends(require_user)):
    job = session.get(ExportJob, export_id)
    if not job:
        raise HTTPException(status_code=404, detail="Export not found")
    if job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return job


@app.post("/eval/report")
def report_eval_status(report: EvalReport, session: Session = Depends(get_session)):
    job = session.get(EvalJob, report.eval_id)
    if not job:
        raise HTTPException(status_code=404, detail="Eval not found")
    job.status = report.status
    job.updated_at = datetime.datetime.utcnow()

    if job.worker_id:
        worker = session.get(Worker, job.worker_id)
        if worker:
            worker.status = WorkerStatus.IDLE
            worker.last_heartbeat = datetime.datetime.utcnow()
            session.add(worker)

    session.add(job)
    session.commit()
    return {"status": "ok"}


@app.post("/export/report")
def report_export_status(report: ExportReport, session: Session = Depends(get_session)):
    job = session.get(ExportJob, report.export_id)
    if not job:
        raise HTTPException(status_code=404, detail="Export not found")
    job.status = report.status
    job.updated_at = datetime.datetime.utcnow()

    if job.worker_id:
        worker = session.get(Worker, job.worker_id)
        if worker:
            worker.status = WorkerStatus.IDLE
            worker.last_heartbeat = datetime.datetime.utcnow()
            session.add(worker)

    session.add(job)
    session.commit()
    return {"status": "ok"}


@app.get("/eval/download/{eval_id}")
def download_eval(eval_id: str, background_tasks: BackgroundTasks, token: Optional[str] = None, authorization: str = Header(default=None), session: Session = Depends(get_session)):
    user: Optional[User] = None
    if token:
        user = get_user_by_token(token)
    elif authorization and authorization.startswith("Bearer "):
        auth_token = authorization.split(" ", 1)[1].strip()
        user = get_user_by_token(auth_token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    job = session.get(EvalJob, eval_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    output_dir = os.path.join(BASE_PATH, "output", eval_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Output not found")
    tmp_base = tempfile.mktemp(prefix=f"{eval_id}_", suffix="")
    zip_path = shutil.make_archive(tmp_base, "zip", root_dir=output_dir)
    background_tasks.add_task(lambda p: os.path.exists(p) and os.remove(p), zip_path)
    return FileResponse(zip_path, media_type="application/zip", filename=f"{eval_id}.zip")


@app.get("/export/download/{export_id}")
def download_export(export_id: str, background_tasks: BackgroundTasks, token: Optional[str] = None, authorization: str = Header(default=None), session: Session = Depends(get_session)):
    user: Optional[User] = None
    if token:
        user = get_user_by_token(token)
    elif authorization and authorization.startswith("Bearer "):
        auth_token = authorization.split(" ", 1)[1].strip()
        user = get_user_by_token(auth_token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    job = session.get(ExportJob, export_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    output_dir = os.path.join(BASE_PATH, "output", export_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Output not found")
    tmp_base = tempfile.mktemp(prefix=f"{export_id}_", suffix="")
    zip_path = shutil.make_archive(tmp_base, "zip", root_dir=output_dir)
    background_tasks.add_task(lambda p: os.path.exists(p) and os.remove(p), zip_path)
    return FileResponse(zip_path, media_type="application/zip", filename=f"{export_id}.zip")


@app.get("/models", response_model=List[ModelRegistry])
def list_models(session: Session = Depends(get_session), user: User = Depends(require_user)):
    return session.exec(select(ModelRegistry).order_by(ModelRegistry.created_at.desc())).all()


@app.post("/models", response_model=ModelRegistry)
def register_model(model: ModelCreate, session: Session = Depends(get_session), user: User = Depends(require_user)):
    existing = session.exec(select(ModelRegistry).where(ModelRegistry.name == model.name)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model already exists")
    record = ModelRegistry(
        name=model.name,
        path=model.path or model.hf_path or model.name,
        hf_path=model.hf_path,
        ms_path=model.ms_path,
        description=model.description,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


@app.post("/inference/deploy")
async def deploy_inference(body: InferenceDeploy, session: Session = Depends(get_session), user: User = Depends(require_user)):
    existing = session.exec(
        select(InferenceSession).where(
            InferenceSession.user_id == user.id,
            InferenceSession.status == InferenceStatus.RUNNING,
        )
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="已有部署，请先卸载")

    deploy_id = f"deploy_{uuid.uuid4().hex[:8]}"

    worker = pick_available_worker(session)
    if not worker:
        raise HTTPException(status_code=503, detail="无可用 Worker")

    config_payload = body.model_dump()
    config_payload["task_id"] = body.task_id
    if not config_payload.get("model_path"):
        config_payload["model_path"] = f"/app/output/{body.task_id}"
    ensure_infer_model_allowed(config_payload.get("model_name"), config_payload.get("model_path"), session, user)

    config_yaml = generate_infer_config(deploy_id, json.dumps(config_payload))

    payload = {
        "deploy_id": deploy_id,
        "user_id": user.id,
        "config_yaml": config_yaml,
        "api_port": body.api_port or 8000,
        "api_host": body.api_host or "0.0.0.0",
        "api_key": config_payload.get("api_key"),
        "api_model_name": config_payload.get("api_model_name"),
        "api_verbose": config_payload.get("api_verbose"),
        "max_concurrent": config_payload.get("max_concurrent"),
        "fastapi_root_path": config_payload.get("fastapi_root_path"),
    }

    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(f"{worker.url.rstrip('/')}/inference/execute", json=payload, timeout=240) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise HTTPException(status_code=resp.status, detail=data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    record = InferenceSession(
        user_id=user.id,
        task_id=body.task_id,
        worker_id=worker.id,
        status=InferenceStatus.RUNNING,
        host=worker.url,
        port=data.get("port", 0),
        container_id=data.get("container_id"),
        updated_at=datetime.datetime.utcnow(),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return {"session_id": record.id, "status": record.status}


@app.post("/inference/stop")
async def stop_inference(body: InferenceStop, session: Session = Depends(get_session), user: User = Depends(require_user)):
    record = session.exec(
        select(InferenceSession).where(
            InferenceSession.user_id == user.id,
            InferenceSession.status == InferenceStatus.RUNNING,
        )
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="无运行中的部署")

    worker = session.get(Worker, record.worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker 未找到")

    payload = {"user_id": user.id}
    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(f"{worker.url.rstrip('/')}/inference/stop", json=payload, timeout=10) as resp:
                if resp.status != 200:
                    data = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    record.status = InferenceStatus.STOPPED
    record.updated_at = datetime.datetime.utcnow()
    session.add(record)
    session.commit()
    return {"status": "stopped"}


@app.post("/inference/chat")
async def chat_inference(body: InferenceChat, session: Session = Depends(get_session), user: User = Depends(require_user)):
    record = session.exec(
        select(InferenceSession).where(
            InferenceSession.user_id == user.id,
            InferenceSession.status == InferenceStatus.RUNNING,
        )
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="无运行中的部署")
    worker = session.get(Worker, record.worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker 未找到")

    payload = {"prompt": body.prompt, "user_id": user.id, "generation_params": body.generation_params}
    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(f"{worker.url.rstrip('/')}/inference/chat", json=payload, timeout=240) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise HTTPException(status_code=resp.status, detail=data)
                return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/task/{task_id}/log")
async def proxy_task_log(websocket: WebSocket, task_id: str, session: Session = Depends(get_session)):
    """Master 作为中间人代理日志：前端 -> Master -> Worker"""
    token = websocket.query_params.get("token")
    user = get_user_by_token(token)
    await websocket.accept()
    if not user:
        await websocket.send_text("Unauthorized")
        await websocket.close()
        return
    task = session.get(Task, task_id)
    if not task or task.user_id != user.id:
        await websocket.send_text("Forbidden")
        await websocket.close()
        return
    try:
        task = session.get(Task, task_id)
        if not task or not task.worker_id:
            await websocket.send_text("Task or worker not found")
            return

        worker = session.get(Worker, task.worker_id)
        if not worker:
            await websocket.send_text("Worker not found")
            return

        # 将 http(s) 转成 ws(s)
        worker_ws_base = worker.url.rstrip("/")
        if worker_ws_base.startswith("https"):
            worker_ws_base = worker_ws_base.replace("https", "wss", 1)
        elif worker_ws_base.startswith("http"):
            worker_ws_base = worker_ws_base.replace("http", "ws", 1)

        worker_ws_url = f"{worker_ws_base}/task/{task_id}/log"

        async with aiohttp.ClientSession() as client:
            async with client.ws_connect(worker_ws_url, heartbeat=20) as worker_ws:
                async for msg in worker_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await websocket.send_text(msg.data)
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        await websocket.send_bytes(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"Proxy error: {e}")
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
