# Master 节点设计文档

## 1. 数据库设计 (models.py)
Master 节点使用 **SQLModel** (基于 SQLAlchemy) + **SQLite** 进行轻量级数据存储。

### 核心表结构
1.  **Worker 表**: 注册的计算节点。
    *   `url`: Worker 的访问地址 (如 `http://localhost:8001`)。
    *   `status`: `IDLE` (空闲), `BUSY` (忙碌), `OFFLINE` (离线)。
    *   `gpu_memory_used/total`: 简单的资源监控字段。
2.  **Task 表**: 训练任务。
    *   `id`: 任务唯一标识 (UUID)。
    *   `status`: `PENDING` -> `RUNNING` -> `SUCCESS`/`FAILED`。
    *   `config_json`: 用户提交的原始参数 (Batch Size, LR, Epoch 等)。
    *   `worker_id`: 关联执行该任务的 Worker。
3.  **User 表**: 鉴权。
    *   简单基于 `api_key` 的验证。默认账号 `admin`, Key `secret-key-123`。

## 2. API 接口规划
*   `POST /submit`: 提交新任务 (Status=PENDING)。
*   `GET /tasks`: 获取任务列表。
*   `POST /worker/register`: Worker 启动时向 Master 注册自己。

## 3. 调度逻辑 (Design)
Master 维护一个后台循环 (Scheduler Loop)：
1.  查询 DB 中所有 `PENDING` 任务。
2.  查询 DB 中所有 `IDLE` 且在线的 Worker。
3.  如果匹配成功：
    *   锁定 Worker (Status -> BUSY)。
    *   生成最终 YAML 配置。
    *   调用 Worker API (`POST /task/execute`)。
    *   更新 Task (Status -> RUNNING)。

## 4. 日志代理 (WebSocket Proxy)
前端无需直接暴露 Worker 地址，可通过 Master 代理日志：
*   前端连接：`ws://<master_host>:8000/ws/task/{task_id}/log`
*   Master 根据 Task 关联的 Worker 组装 `ws://<worker>/task/{task_id}/log`，并用 aiohttp 转发消息。
*   支持 http/https -> ws/wss 自动转换；文本日志透传。
