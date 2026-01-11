#!/usr/bin/env bash
set -euo pipefail

# 中文注释：项目根目录，可用 BASE_DIR 环境变量覆写，避免硬编码路径
BASE_DIR="${BASE_DIR:-/home/ubuntu/CloudComputing}"
# 中文注释：可自定义启动命令，默认为 python3 -m master.main / worker.main
MASTER_CMD="${MASTER_CMD:-python3 -m master.main}"
WORKER_CMD="${WORKER_CMD:-python3 -m worker.main}"

usage() {
  echo "用法: $0 <start|stop|restart> <master|worker|both> [--master-args \"...\"] [--worker-args \"...\"]"
  echo "示例: $0 start both --master-args \"--port 8000\" --worker-args \"--port 8001\""
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

ACTION="$1"
TARGET="$2"
shift 2

MASTER_ARGS=""
WORKER_ARGS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --master-args)
      MASTER_ARGS="$2"; shift 2 ;;
    --worker-args)
      WORKER_ARGS="$2"; shift 2 ;;
    *)
      echo "未知参数: $1"; usage; exit 1 ;;
  esac
done

stop_master() {
  # 中文注释：优雅关闭 master
  pkill -f "python3 -m master.main" 2>/dev/null || true
}

stop_worker() {
  # 中文注释：优雅关闭 worker
  pkill -f "python3 -m worker.main" 2>/dev/null || true
}

start_master() {
  cd "$BASE_DIR"
  # 中文注释：前台命令改为 nohup 后台运行，日志落盘 master.log
  nohup $MASTER_CMD $MASTER_ARGS > master.log 2>&1 &
  echo "master 启动，PID=$!"
}

start_worker() {
  cd "$BASE_DIR"
  # 中文注释：前台命令改为 nohup 后台运行，日志落盘 worker.log
  nohup $WORKER_CMD $WORKER_ARGS > worker.log 2>&1 &
  echo "worker 启动，PID=$!"
}

case "$ACTION" in
  start)
    if [[ "$TARGET" == "master" || "$TARGET" == "both" ]]; then
      start_master
    fi
    if [[ "$TARGET" == "worker" || "$TARGET" == "both" ]]; then
      start_worker
    fi
    ;;
  stop)
    if [[ "$TARGET" == "master" || "$TARGET" == "both" ]]; then
      stop_master
    fi
    if [[ "$TARGET" == "worker" || "$TARGET" == "both" ]]; then
      stop_worker
    fi
    ;;
  restart)
    if [[ "$TARGET" == "master" || "$TARGET" == "both" ]]; then
      stop_master
      start_master
    fi
    if [[ "$TARGET" == "worker" || "$TARGET" == "both" ]]; then
      stop_worker
      start_worker
    fi
    ;;
  *)
    echo "未知动作: $ACTION"; usage; exit 1 ;;
 esac
