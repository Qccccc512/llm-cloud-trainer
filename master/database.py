from typing import List
from sqlmodel import SQLModel, create_engine, Session, select, delete
from master.models import User, ModelRegistry, Dataset
import os
import json
import datetime

# 允许通过环境变量配置项目根与共享目录，便于 NFS/多机部署
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
CLOUD_LLM_ROOT = os.environ.get("CLOUD_LLM_ROOT", os.path.join(PROJECT_ROOT, "cloud-llm"))
REFERENCE_DIR = os.environ.get("REFERENCE_DIR", os.path.join(PROJECT_ROOT, "reference"))

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=False)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(CLOUD_LLM_ROOT, "data"))
DATASET_INFO_PATH = os.path.join(DATA_DIR, "dataset_info.json")
DATASET_INFO_PUBLIC_PATH = os.path.join(DATA_DIR, "dataset_info_public.json")


def ensure_modelregistry_columns():
    """尽力为已存在的 SQLite 数据库补充 hf_path/ms_path 字段。"""
    try:
        with engine.connect() as conn:
            cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(modelregistry)").fetchall()}
            altered = False
            for col in ("hf_path", "ms_path"):
                if col not in cols:
                    conn.exec_driver_sql(f"ALTER TABLE modelregistry ADD COLUMN {col} VARCHAR;")
                    altered = True

            # 补充描述与时间戳字段，便于排序与展示
            timestamp_cols = {
                "description": "VARCHAR",
                "created_at": "DATETIME",
                "updated_at": "DATETIME",
            }
            for col, typ in timestamp_cols.items():
                if col not in cols:
                    conn.exec_driver_sql(f"ALTER TABLE modelregistry ADD COLUMN {col} {typ};")
                    altered = True

            if altered:
                # 为缺省时间戳补齐当前时间，避免 NULL 导致排序/序列化异常
                now = datetime.datetime.utcnow().isoformat(sep=" ")
                conn.exec_driver_sql("UPDATE modelregistry SET created_at = COALESCE(created_at, ?), updated_at = COALESCE(updated_at, ?);", (now, now))
                conn.commit()
    except Exception:
        # 避免阻塞启动，必要时可重新执行迁移
        pass


def ensure_worker_columns():
    """为已存在的 worker 表补充 CPU/内存监控字段。"""
    try:
        with engine.connect() as conn:
            cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(worker)").fetchall()}
            altered = False
            for col, typ in ("cpu_percent", "FLOAT"), ("mem_percent", "FLOAT"):
                if col not in cols:
                    conn.exec_driver_sql(f"ALTER TABLE worker ADD COLUMN {col} {typ};")
                    altered = True
            if altered:
                conn.commit()
    except Exception:
        pass


def _load_json_file(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _infer_dataset_source(info: dict) -> (str, str):
    """根据 dataset_info 条目推断来源与路径。返回 (source, path)。"""
    if not isinstance(info, dict):
        return "unknown", None
    if info.get("file_name"):
        return "local", f"/app/data/{info.get('file_name')}"
    if info.get("hf_hub_url"):
        return "hf", info.get("hf_hub_url")
    if info.get("ms_hub_url"):
        return "ms", info.get("ms_hub_url")
    if info.get("cloud_file_name"):
        return "cloud", info.get("cloud_file_name")
    return "unknown", None


def upsert_public_datasets(session: Session):
    """将 dataset_info_public.json 中的条目注册为公共数据集。"""
    public_info = _load_json_file(DATASET_INFO_PUBLIC_PATH)
    if not public_info:
        return
    changed = False
    for name, info in public_info.items():
        source, path = _infer_dataset_source(info)
        existing = session.exec(select(Dataset).where(Dataset.name == name, Dataset.visibility == "public")).first()
        if existing:
            # 更新 info/path/remark
            need_update = False
            if existing.info != info:
                existing.info = info
                need_update = True
            if existing.path != path:
                existing.path = path
                need_update = True
            if existing.source != source:
                existing.source = source
                need_update = True
            if need_update:
                existing.updated_at = datetime.datetime.utcnow()
                session.add(existing)
                changed = True
            continue

        ds = Dataset(
            name=name,
            display_name=name,
            visibility="public",
            source=source,
            path=path,
            info=info,
            remark="公共预置数据集",
        )
        session.add(ds)
        changed = True
    if changed:
        session.commit()


def sync_dataset_info_json(session: Session):
    """将数据库中的数据集同步写入 cloud-llm/data/dataset_info.json。"""
    datasets = session.exec(select(Dataset)).all()
    data = {}
    for ds in datasets:
        info = ds.info if isinstance(ds.info, dict) else _load_json_file("/dev/null")
        if not isinstance(info, dict):
            info = {}
        # 为上传数据集自动补齐 file_name
        if ds.source == "upload" and not info.get("file_name") and ds.path:
            rel = ds.path
            if rel.startswith("/app/data/"):
                rel = rel[len("/app/data/"):].lstrip("/")
            info["file_name"] = rel
        data[ds.name] = info

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_model_mappings() -> dict:
    mapping_path = os.environ.get("MODEL_MAPPING_PATH", os.path.join(REFERENCE_DIR, "extracted_model_paths.json"))
    if not os.path.isfile(mapping_path):
        return {}
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def refresh_model_registry(names: List[str], mapping: dict, rebuild: bool = False):
    """根据 reference/extracted_model_paths.json 刷新 ModelRegistry；可选强制重建。"""
    with Session(engine) as session:
        if rebuild:
            session.exec(delete(ModelRegistry))
            session.commit()

        existing_map = {m.name: m for m in session.exec(select(ModelRegistry)).all()}
        changed = False
        for n in names:
            hf = mapping.get(n, {}).get("hf")
            ms = mapping.get(n, {}).get("ms")
            if n in existing_map:
                record = existing_map[n]
                updated = False
                if hf and record.hf_path != hf:
                    record.hf_path = hf
                    updated = True
                if ms and record.ms_path != ms:
                    record.ms_path = ms
                    updated = True
                if hf and record.path != hf:
                    record.path = hf
                    updated = True
                if updated:
                    session.add(record)
                    changed = True
            else:
                session.add(ModelRegistry(name=n, path=hf or n, hf_path=hf, ms_path=ms, description="预置模型"))
                changed = True
        if changed:
            session.commit()

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

def init_db_data():
    """Initialize default user or worker if needed"""
    ensure_modelregistry_columns()
    ensure_worker_columns()
    # 初始化公共数据集到数据库，并同步 dataset_info.json
    model_mappings = load_model_mappings()
    rebuild_registry = os.getenv("MODEL_REGISTRY_RESET", "").lower() in {"1", "true", "yes"}
    with Session(engine) as session:
        # Check if default admin exists
        user = session.query(User).filter(User.username == "admin").first()
        if not user:
            import hashlib, secrets
            def _hash_pwd(p: str) -> str:
                salt = secrets.token_hex(8)
                digest = hashlib.sha256((salt + p).encode("utf-8")).hexdigest()
                return f"{salt}${digest}"
            admin = User(username="admin", api_key="secret-key-123", password_hash=_hash_pwd("admin123"))
            session.add(admin)
            session.commit()

        # 初始化空的模型注册表示例（可选）
        existing_model = session.query(ModelRegistry).first()
        if not existing_model:
            demo = ModelRegistry(name="demo-model", path="/app/output/demo", description="示例占位模型")
            session.add(demo)
            session.commit()

        # 预置模型列表：基于 reference/extracted_model_paths.json 的键值初始化（仅 HF 下载）。
        try:
            names = list(model_mappings.keys()) if model_mappings else []
            if names:
                refresh_model_registry(names, model_mappings, rebuild_registry)
            upsert_public_datasets(session)
            sync_dataset_info_json(session)
        except Exception:
            # 避免初始化失败影响启动
            session.rollback()
            pass
