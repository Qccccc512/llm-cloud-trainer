from typing import Optional, List
from sqlmodel import Field, SQLModel, Relationship
from datetime import datetime
from enum import Enum
from sqlalchemy import UniqueConstraint, Column, JSON

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

class WorkerStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


class InferenceStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"


class EvalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class ExportStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

class Worker(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(index=True, unique=True, description="Worker 内网/优先访问 URL")
    status: WorkerStatus = Field(default=WorkerStatus.IDLE)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    
    # CPU/内存监控（百分比）
    cpu_percent: float = 0.0
    mem_percent: float = 0.0

    # 显存监控 (MB)
    gpu_memory_used: int = 0
    gpu_memory_total: int = 24576
    gpu_utilization: int = 0
    gpu_temperature: int = 0
    
    tasks: List["Task"] = Relationship(back_populates="worker")

class Task(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True) # 使用 UUID
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # 核心配置
    config_json: str = Field(description="JSON format of training parameters")
    dataset_path: str = Field(default="/app/data", description="Path to dataset inside container")
    
    # 关联
    worker_id: Optional[int] = Field(default=None, foreign_key="worker.id")
    worker: Optional[Worker] = Relationship(back_populates="tasks")
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    user: Optional["User"] = Relationship()


class EvalJob(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    status: EvalStatus = Field(default=EvalStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # 配置
    config_json: str = Field(description="JSON format of eval parameters")
    dataset_path: str = Field(default="/app/data", description="Path to dataset inside container")

    # 关联
    worker_id: Optional[int] = Field(default=None, foreign_key="worker.id")
    worker: Optional[Worker] = Relationship()
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    user: Optional["User"] = Relationship()


class ExportJob(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    status: ExportStatus = Field(default=ExportStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    config_json: str = Field(description="JSON format of export parameters")
    dataset_path: str = Field(default="/app/data", description="Path to dataset inside container")

    worker_id: Optional[int] = Field(default=None, foreign_key="worker.id")
    worker: Optional[Worker] = Relationship()
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    user: Optional["User"] = Relationship()


class InferenceSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    task_id: str
    worker_id: int = Field(foreign_key="worker.id")
    status: InferenceStatus = Field(default=InferenceStatus.RUNNING)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=0)
    container_id: Optional[str] = Field(default=None)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ResourceMetric(SQLModel, table=True):
    """Worker 资源监控日志，便于后续做可视化曲线。"""
    id: Optional[int] = Field(default=None, primary_key=True)
    worker_id: int = Field(foreign_key="worker.id")
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    cpu_percent: float = 0.0
    mem_percent: float = 0.0
    gpu_utilization: int = 0
    gpu_memory_used: int = 0
    gpu_memory_total: int = 0
    gpu_temperature: int = 0

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True)
    api_key: str = Field(index=True)
    password_hash: str = Field(default="")


class ModelRegistry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True, description="模型名称或标识")
    path: str = Field(description="模型或适配器在共享存储中的路径")
    hf_path: Optional[str] = Field(default=None, description="Hugging Face 模型仓库标识")
    ms_path: Optional[str] = Field(default=None, description="ModelScope 模型仓库标识")
    description: Optional[str] = Field(default=None, description="模型备注")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Dataset(SQLModel, table=True):
    __tablename__ = "dataset"
    __table_args__ = (UniqueConstraint("name", "owner_id", name="uq_dataset_owner_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, description="数据集标识，与 dataset_info.json 的键一致")
    display_name: Optional[str] = Field(default=None, description="展示名称")
    visibility: str = Field(default="public", description="public 或 private")
    source: str = Field(default="local", description="local/hf/ms/url/upload")
    path: Optional[str] = Field(default=None, description="数据集在 /app/data 下的相对或绝对路径，或 hub 标识")
    info: dict = Field(default_factory=dict, sa_column=Column(JSON))
    remark: Optional[str] = Field(default=None, description="备注说明")
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = Field(default=None)
