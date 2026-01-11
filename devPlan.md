这份开发文档旨在为你提供一个清晰、可执行的工程落地指南。我们将项目拆分为**基础设施准备**、**核心后端开发**、**Worker开发**、**前端开发**以及**联调与优化**五个阶段。

---

# 云端大模型训练平台（LLM-Cloud-Trainer）开发文档

> 稳定版声明：本文档以当前代码库实现为 **稳定版 v1（Stable v1）** 的功能边界与使用说明；
> 所有“未实现/待办/后续规划”统一收敛到文末 **“有待进一步完善”**，避免与现状混杂。

## 1. 系统架构概览

在单机模拟分布式的环境下，我们将逻辑架构划分为三层：

*   **接入层 (Frontend)**: 用户交互界面，负责配置参数、查看日志、下载模型、**在线对话测试**。
*   **控制层 (Master Node)**: 系统的大脑。负责鉴权、任务调度、生成配置、状态管理、WebSocket日志转发、**模型仓库管理**。
*   **计算层 (Worker Node)**: 系统的手脚。负责接收指令、拉起 Docker 容器、挂载数据、执行训练/评估/推理、上报心跳与**GPU监控**。

**通信协议**：
*   Master <-> Worker: HTTP (REST API) + WebSocket (Log Stream)
*   Master <-> Frontend: HTTP (REST API) + WebSocket (Log Stream)
*   Master <-> Docker Daemon: Docker SDK for Python
*   **存储层 (Future)**: NFS (Network File System) 实现多节点间的数据集与产物共享。

**近期更新（当前迭代）**
- Worker 路径/缓存/日志目录支持环境变量覆写；推理容器落盘日志并可 HTTP 下载；后台清理扩展至所有 `llm-*` 容器。
- Heartbeat 上报 CPU/内存+GPU 指标，自动注册 Worker 并记录 ResourceMetric；60s 标记 OFFLINE，24h 清理僵尸注册。
- 新增 `scripts/manage_services.sh` 启停/重启 master/worker。
- 导出/推理 YAML：默认 bitsandbytes，GPTQ 需显式选择；推理 YAML 透传生成参数。
- **数据集管理**：Dataset 表+公共数据集同步+私有上传/注册/备注/删除；数据库变更写回 `dataset_info.json`；前端公共集隐藏删除/备注。
- **模型来源/安全收敛**：前后端限制训练/评测/部署/导出仅用预置或本人产物；阻止 LoRA 量化“先合并导出再量化”；推理返回隐藏 host/port；部署页隐藏模型路径。
- **训练表单增强**：阶段化展示；预训练强制全参+train_from_scratch；LoRA 支持 QLoRA（bit/method/type/double）；通用 warmup/cutoff、验证集比例；Eval Steps 提示需配 evaluation_strategy=steps；额外参数 JSON 占位示例。
- **模型选择约束**：训练仅预置+导出；评测/部署/导出支持预置+训练成功+导出成功；前端自动默认选项并自动刷新 Worker 状态。

---

## 2. 技术选型建议

*   **语言**: Python 3.10+ (因涉及大量 AI 配置处理，Python 最顺手)
*   **Web 框架 (Master/Worker)**: **FastAPI** (高性能，原生支持异步 WebSocket，适合处理长连接)
*   **数据库**: **SQLite** (基础版够用，无需部署额外服务) 或 **MySQL** (进阶推荐)
*   **容器控制**: **Docker SDK for Python** (`pip install docker`)
*   **核心引擎**: **LLaMA-Factory** (Docker 镜像)
*   **前端**: Vue 3 / React + **Xterm.js** (用于展示像终端一样的实时日志)

---

## 3. 稳定版范围（已实现功能清单）

### 基础设施 (Infrastructure)
- [x] 服务器环境配置 (NVIDIA Driver, CUDA, Docker, NVIDIA Container Toolkit)（PoC 环境已验证）
- [x] 构建/拉取 LLaMA-Factory 基础镜像并固化版本（当前 Worker 默认镜像为 `lf-with-optimum:v1.0`）
- [x] 规划宿主机共享目录结构（当前已使用：`./cloud-llm/{data, output, logs, checkpoints, hf-cache, temp}`）

### Master 节点 (Control Plane)
- [x] 数据库模型设计 (Task, Worker, User, **ModelRegistry**)
- [x] 数据集表设计与权限模型（Dataset：public/private + owner_id）
- [x] API: 接收用户训练请求，校验参数
- [x] Logic: 任务调度队列 (Queue)
- [x] Logic: 基础 YAML 模板 + 深度合并（支持 `config_overrides`）生成训练/评估/导出/推理配置
- [x] API: WebSocket 聚合日志流
- [x] **Logic: 推理服务调度 (启动模型的 API 容器，超时时间与 Worker 对齐 240s)**
- [x] 模型注册表：支持 hf_path/ms_path 字段；从 `reference/extracted_model_paths.json` 初始化；提交任务/评估/导出时基于名称映射为 HF 路径
- [x] 模型注册表修复：补齐 created_at/updated_at 字段，`GET /models` 排序不再报错
- [x] 数据集管理 API：`GET /datasets`（未登录可浏览公共，登录可见私有）、`POST /datasets/upload`、`POST /datasets/register`、`PATCH /datasets/{id}/remark`、`DELETE /datasets/{id}`
- [x] 数据集信息同步：DB <-> `cloud-llm/data/dataset_info.json`（训练/评测容器挂载 `/app/data` 使用）

### Worker 节点 (Data Plane)
- [x] API: 接收 Master 下发的任务指令 (Train/Infer)
- [x] Logic: 调用 Docker SDK 启动容器 (训练容器 vs 推理容器，推理容器支持 LoRA + base auto 推断)
- [x] Logic: 实时读取容器日志并通过 WebSocket/HTTP 推送给 Master
- [x] Logic: 异常捕获 (OOM 检测) 与 容器清理
- [x] **Logic: 获取 GPU 实时状态 (显存/利用率) 并随心跳上报**
- [x] 推理容器 HTTP 就绪探测（TCP + /v1/chat/completions，等待 180s）；挂载 HF cache `/home/ubuntu/CloudComputing/cloud-llm/hf-cache` 到容器 `/root/.cache/huggingface` 以复用权重下载
- [x] 推理部署改为 `/inference/execute` + YAML 驱动，支持 `config_overrides` 与生成参数透传；API 元信息以环境变量传递
- [x] 导出配置：默认 bitsandbytes 量化；非 GPTQ 时自动移除 `export_quantization_*`，避免缺少 GPTQ 依赖时报错
- [x] Worker 路径/日志/HF cache 支持环境变量覆写；推理日志落盘与下载接口；所有 `llm-*` 容器退出后自动清理

### 前端 (User Interface)
- [x] 页面: 训练参数配置表单（阶段化选项、LoRA/QLoRA、train_from_scratch、warmup/cutoff、验证集比例、额外参数 JSON）
- [x] 页面: 任务列表与状态看板 (Pending, Running, Finished)
- [x] 组件: 基于 WebSocket 的实时日志控制台（通过 Master `/ws/task/{task_id}/log` 代理）
- [x] **页面: 模型在线体验 (Playground) - 对话框组件（部署/卸载/对话，状态徽章，本地会话持久化）**
- [x] 数据集管理页面：上传/列表/备注/删除（公共数据集限制删除/修改）
- [x] 训练/评测数据集选择：使用后端 `/datasets` 结果渲染下拉选项（页面加载即拉取公共数据集）
- [x] 模型选择控件：前端限制来源（预置/本人产物），提交前后端双重校验，默认选项自动填充

---

## 4. 分阶段详细开发规划

### 第一阶段：基础设施与 PoC (Proof of Concept)
**目标**：不写 Web 代码，手动验证 Docker 训练流程，确保底层通畅。

1.  **环境准备**：
    *   **[Completed]** 在 A10 服务器上安装 Docker 和 NVIDIA Container Toolkit。
    *   **[Completed]** 运行 `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi` 确保容器能看到 GPU。
2.  **镜像准备**：
    *   **[Completed]** 拉取/构建 LLaMA-Factory 镜像并固化版本（当前默认：`lf-with-optimum:v1.0`）。
3.  **目录挂载验证**：
    *   **[Completed]** 创建宿主机目录：`./cloud-llm/data` (放一个 `dataset_info.json` 和简单的 json 数据集)。
    *   **[Completed]** 创建宿主机目录：`./cloud-llm/output/test_task`。
4.  **手动训练测试**：
    *   **[Completed]** 编写 `test_config.yaml`。
    *   **[Completed]** 执行 Docker 命令（模拟 Worker 行为）并验证成功。
    *   **[Completed]** 确认模型能跑通，产物落在宿主机共享目录。

### 第二阶段：Worker 开发 (核心执行器)
**目标**：把第一阶段的手动命令封装成 HTTP 服务。

1.  **Worker 服务搭建** (FastAPI)：
    *   **[Completed]** `POST /task/execute`: 接收 Master 发来的 Payload (包含 yaml 内容、TaskID)。
    *   **[Completed]** 启动后自动挂载共享目录（默认 `./cloud-llm`，支持 `HOST_BASE_PATH/LOG_BASE_PATH/HF_CACHE_PATH` 环境变量覆写）。
2.  **Docker SDK 集成**：
    *   **[Completed]** 接收到请求后，将 yaml 内容写入宿主机的临时文件。
    *   **[Completed]** 使用 `client.containers.run(..., detach=True)` 启动容器。
    *   **[Completed]** 设置 `shm_size="2g"`。
3.  **日志流处理**：
    *   **[Completed]** 开发 `WS /task/{task_id}/log`: 通过 WebSocket 持续向连接者发送日志行。
4.  **状态监控与清理**：
    *   **[Completed]** 开发后台线程，每30秒轮询并删除 `Exited` 状态的 `llm-worker-*` 容器，并增加延迟清理避免过早删除。
    *   **[Completed]** 心跳上报 GPU 显存/利用率/温度；容器退出回调上报 OOM 提示。
    *   **[Completed]** 推理容器支持（/inference/start,/stop,/chat）已上线，含 TCP+HTTP 就绪探测、HF cache 挂载、空闲自动卸载与日志回报。

### 第三阶段：Master 开发 (调度与业务逻辑)
**目标**：管理任务状态，连接前端与 Worker。

1.  **数据库设计**：
    *   Table `Tasks`: `id`, `status` (pending/running/success/failed), `config_json`, `created_at`, `worker_ip`.
    *   Table `Workers`: `ip`, `status` (idle/busy), `last_heartbeat`.
2.  **配置生成模块**：
    *   使用 `master/templates/*.yaml` 作为基础模板。
    *   将前端表单参数映射到少量关键字段，并通过 `config_overrides` 深度合并覆写，生成最终 YAML（训练/评测/导出/推理）。
3.  **任务调度器**：
    *   **[Completed]** 当用户提交任务 -> 存入 DB (Pending)。
    *   **[Completed]** 调度循环：查找 Status=Pending 的任务 AND Status=Idle 的 Worker -> 锁定 Worker -> 发送请求给 Worker -> 更新任务状态 (Running)。
    *   **[Completed]** Worker 回调 `/task/report` + Master 兜底轮询，收敛为 success/failed 并释放 Worker。
4.  **日志转发 (Proxy)**：
    *   前端连接 Master 的 WS -> Master 连接 Worker 的 WS -> 转发数据。（这样前端不需要知道 Worker 的 IP，便于未来做内网隔离）。
5.  **产物下载**：
    *   **[Completed]** Master 提供 `GET /download/{task_id}` 打包输出目录为 zip。
6.  **模型注册表**：
    *   **[Completed]** 增加 ModelRegistry 表及 GET/POST API。
    *   **[Completed]** 修复 `/models` 排序报错：补齐 created_at/updated_at 字段并完成迁移。
7.  **用户与鉴权**：
    *   **[Completed]** 支持注册（邀请码校验）/登录，Bearer Token 鉴权保护用户接口。
    *   **[Completed]** 任务与用户绑定，列表/详情/日志/下载仅可访问本人任务。
8.  **在线推理调度**：
    *   **[Completed]** Master/Worker 支持推理容器部署、停止、对话转发，一人仅一活跃部署；5 分钟无请求自动卸载。

9.  **数据集管理（新增）**：
    *   **[Completed]** Dataset 表：public/private + owner_id，支持备注与元信息（info JSON）。
    *   **[Completed]** 启动同步：从 `cloud-llm/data/dataset_info_public.json` 注册公共数据集到数据库。
    *   **[Completed]** 同步写回：数据库数据集变更自动写入 `cloud-llm/data/dataset_info.json`。
    *   **[Completed]** 权限控制：公共集默认只读；私有集仅本人可改/删。

### 第四阶段：前端开发与联调
**目标**：可视化的操作界面。

1.  **参数配置页**：
    *   **简化原则**：不要暴露 LLaMA-Factory 的所有参数。
    *   提供核心选项：Base Model (下拉选择), Dataset, Learning Rate, Epochs, LoRA Rank, Batch Size (建议根据显存写死最大值或提供安全范围)。
    *   **[Completed]** 页面已经实现上述输入控件，并通过 `/submit` 向 Master 传递训练参数。
    *   **[Completed]** 数据集选择使用后端 `/datasets` 的结果渲染下拉（select），不再依赖 datalist（避免仅显示 identity）。
2.  **任务列表页**：
    *   轮询获取任务状态列表。
    *   点击“查看日志” -> 打开 WebSocket 弹窗，接入 Master 的日志流。
    *   **[Completed]** 对 success 任务提供下载按钮，直连 `/download/{task_id}`。
    *   **[Completed]** 任务状态过滤、任务详情（配置 JSON 展示、日志入口、下载入口）。
    *   **[Completed]** 表格已接 `/tasks`、按状态过滤、打开日志 WebSocket、拉起详情弹窗与下载。
3.  **Worker 监控视图**：
    *   **[Completed]** 表格展示 Worker 状态、GPU 显存/利用率/温度、心跳时间。
4.  **模型在线体验 (Playground)**：
    *   **[Completed]** 部署/停止/对话入口，聊天记录本地保存，状态徽章显示端口；后端推理接口鉴权已接入。
5.  **登录/注册前端**：
    *   **[Completed]** 登录/注册表单，持久化 token，向 REST/WS 附带鉴权。
6.  **任务隔离展示**：
    *   **[Completed]** 前端任务列表、日志、下载入口按用户 token 限定。
7.  **推理入口**：
    *  **[Completed]** 推理卡片已经支持部署/停止/对话、状态徽章与本地状态恢复。
8.  **邀请码机制**：
    *   **[Completed]** 注册时在 `/auth/register` 请求体中提交 `invite_code`，由 Master 按 `INVITE_CODE` 环境变量校验；前端注册弹窗已提供输入框。

### 第五阶段：部署与推理服务 (Pro Max 特性)
**目标**：不仅能练，还能用。实现训练产物的在线加载和对话。

1.  **Worker 端推理支持**：
    *   **[Completed]** `/inference/start`/`/stop`/`/chat` 已实现；支持 LoRA + base auto 推断；端口映射随机分配。
    *   **[Completed]** 启动就绪探测 (TCP + HTTP)，避免服务未加载时聊天失败。
    *   **[Completed]** HF cache 挂载复用权重，减少重复下载。
2.  **前端 Playground**：
    *   **[Completed]** 推理入口独立卡片，部署/停止/对话全链路接 Master 转发到 Worker。

### 第六阶段：模型评估服务
**目标**：在训练后提供标准化评估能力，形成可比较的指标与报告。

1.  **[Completed]** Worker 评估执行：新增 `/eval/execute`，当前实现使用 `llamafactory-cli train` 搭配评估 YAML（由模板控制 `do_eval` 等），写临时 YAML、挂载数据与 HF cache，日志落盘并回报状态。
2.  **[Completed]** Master 评估调度与结果存储：新增 `/eval/submit`/`/evals`/`/eval/{id}`/`/eval/report`/`/eval/download/{id}`，提交即分发到空闲 Worker，状态机与训练一致并可下载输出。
3.  **[Completed]** 前端评估视图：新增评估卡片，支持填写模型路径/名称、数据集、模板、batch/max_samples，查看评估列表并下载产物。

### 第七阶段：模型导出（量化/合并后导出）
**目标**：将训练/适配后的模型做量化或合并，并导出可直接部署的产物。

1.  **[Completed]** Worker 导出流程：支持调用 `llamafactory-cli export`，参数包含量化位宽（如 4/8 bit）、量化数据集与输出路径；共享目录下生成导出包。
2.  **[Completed]** Master 导出任务管理：已提供导出任务接口与状态追踪（`/export/submit`、`/exports`、`/export/{id}`、`/export/report`、`/export/download/{id}`）。
3.  **[Completed]** 前端导出与下载：已提供导出卡片，支持提交导出、查看列表与一键下载。
5.  **现状备注**：默认走 bitsandbytes 流程以规避 GPTQ 依赖缺失；如需 GPTQ，需显式选择并为镜像安装相关依赖；LoRA 量化仍维持“先合并导出再量化”的前端限制（待补自动合并/两段式流程）。

> 注：所有“未实现/待进一步完善”的需求已统一移动到文末 **“有待进一步完善”** 章节。

---

## 5. 自测清单（建议按顺序执行，便于快速定位问题）

### 5.1 服务与数据库
- 数据库文件以仓库根目录的 `database.db` 为准（当前线上运行使用该文件），历史遗留的 `master/database.db` 不再作为主库。
- 访问 `GET /datasets`（不带 token）应返回公共数据集列表；登录后（带 Bearer token）应额外返回自己的私有数据集。
- 访问 `GET /models` 不应再出现 created_at 报错。

### 5.2 前端关键链路
- 打开首页后（未登录也可）训练/评测的数据集下拉应能看到多条公共数据集（不应仅 identity）。
- 登录后上传私有数据集：刷新列表后可见；训练/评测下拉应出现该私有数据集。
- 私有数据集删除后：列表与下拉均不再出现；`cloud-llm/data/dataset_info.json` 同步更新。

---

## 6. 关键代码逻辑示例 (Python)

### 1. 配置生成：YAML 模板 + 深度合并
当前实现不再依赖 Jinja2 模板渲染，而是使用 `master/templates/*.yaml` 作为基础模板，结合前端表单与 `config_overrides` 做深度合并生成最终 YAML：

- 训练：`master/templates/train_base.yaml` + `generate_config()`
- 评测：`master/templates/eval_base.yaml` + `generate_eval_config()`
- 导出：`master/templates/export_base.yaml` + `generate_export_config()`
- 推理：`master/templates/inference_base.yaml` + `generate_infer_config()`

合并逻辑的核心原则：模板提供默认值；前端表单映射到少量关键字段；其余高级参数通过 `config_overrides` 原样覆写。

### 2. Worker 启动 Docker（简化示意，按当前实现）
```python
import docker
import os

client = docker.from_env()

def start_training(task_id: str, config_content: str):
    # 1. 写入配置文件到宿主机共享目录
    # 注意：这里的路径是宿主机（Worker）上的路径；实际目录支持环境变量覆写
    host_base = os.environ.get("HOST_BASE_PATH", os.path.abspath("./cloud-llm"))
    host_config_path = f"{host_base}/temp/{task_id}.yaml"
    os.makedirs(os.path.dirname(host_config_path), exist_ok=True)
    with open(host_config_path, "w") as f:
        f.write(config_content)
    
    # 获取绝对路径用于 Docker 挂载
    base_path = host_base

    hf_cache = os.environ.get("HF_CACHE_PATH", f"{base_path}/hf-cache")

    # 2. 启动容器
    container = client.containers.run(
        image="lf-with-optimum:v1.0",
        command=f"llamafactory-cli train /app/temp/{task_id}.yaml",
        device_requests=[ docker.types.DeviceRequest(count=-1, capabilities=[['gpu']]) ],
        volumes={
            f'{base_path}/data': {'bind': '/app/data', 'mode': 'ro'},
            f'{base_path}/output': {'bind': '/app/output', 'mode': 'rw'},
            # 挂载宿主机存放配置文件的目录到容器内
            f'{base_path}/temp': {'bind': '/app/temp', 'mode': 'ro'},
            # 复用 HuggingFace 缓存，减少重复下载
            hf_cache: {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
        },
        detach=True,
        shm_size='2g',
        environment=["HF_ENDPOINT=https://hf-mirror.com"],
        name=f"llm-worker-{task_id}",
    )
    return container.id
```

## 7. 进阶提示

*   **模型下载加速与复用**：已挂载宿主 `/home/ubuntu/CloudComputing/cloud-llm/hf-cache` 到容器 `/root/.cache/huggingface`，减少重复下载。后续可预拉常用 Base 模型到该缓存。
*   **长耗时推理就绪**：Master 调用 Worker 推理接口超时已放宽至 240s，Worker 内部就绪等待 180s，适配首次权重下载。

*   **数据集注册与同步**：LLaMA-Factory 依赖 `dataset_info.json` 识别数据集。
    *   **当前实现**：通过 Master 的数据集管理接口（上传/注册/删除/备注）维护 Dataset 表，并在变更时同步写回 `cloud-llm/data/dataset_info.json`，供训练/评测容器直接使用。
*   **A10 显存限制**：
    *   在前端限制 Batch Size 最大为 4 或 8。
    *   强制开启 4bit 量化 (QLoRA) 和 梯度检查点 (Gradient Checkpointing)，否则 7B 模型容易 OOM。

按照这个文档执行，你的大作业结构将非常清晰，且具备极高的工程完成度。

---

## 8. 有待进一步完善（未实现需求汇总）

以下内容 **不属于稳定版 v1**，建议作为后续迭代清单统一推进。

### 8.1 基础设施与部署
- NFS/多机共享存储：将 `cloud-llm/{data,output,logs,checkpoints,hf-cache,temp}` 迁移到可共享挂载点，并复核容器挂载一致性。
- 镜像与依赖固化：为推理/训练/评测/导出准备更明确的镜像变体与依赖矩阵（尤其是 GPTQ 相关）。

### 8.2 Master（控制面）
- Worker 离线时运行中任务的降级/重试策略。
- 调度公平性与多任务并发分发（当前为简化的单任务/可用 worker 选择）。
- 资源日志保留策略：`ResourceMetric` 的体积控制（按时间窗口清理/归档）与查询接口。
- 模型注册表管理增强：前端可编辑/双向校验；补充 ModelScope 下载支持。
- 推理资源占用监控与前端管理入口（例如展示 session 占用、并发限制、在线时长等）。
- 导出产物自动写入 ModelRegistry：在 `/export/report` 成功时把 `/app/output/<export_id>/export` 注册进模型库。
- 安全收敛：`/workers` 返回内容脱敏（避免暴露内部 `url`），前端仅展示必要的资源指标与状态。
- GPTQ 依赖可选化：提供依赖检测/提示或独立镜像；默认 bitsandbytes 流程不应触发 GPTQ 解析。

### 8.3 Worker（数据面）
- 在镜像内补齐 `psutil`，避免 CPU/内存监控在未安装时降级为 0。
- OOM 处理闭环增强：Exit Code/日志关键字识别与更明确的失败原因上报。

### 8.4 前端（体验与可观测）
- Worker 异常/离线告警提示与筛选。
- Worker 资源曲线展示（CPU/内存/GPU）与历史回放。
- 推理空闲倒计时提醒（与 Worker 现有 5 分钟自动卸载逻辑联动）。
- 数据集列表搜索/分页（数据集条目多时提升可用性）。
- 评估指标规范与可视化：约定指标字段（bleu/rouge/accuracy/latency 等），在前端展示图表（当前可先下载输出查看）。