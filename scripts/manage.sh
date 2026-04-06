#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKEND_PID_FILE="${ROOT_DIR}/server.pid"
BACKEND_LOG_FILE="${ROOT_DIR}/server.log"
FRONTEND_PID_FILE="${ROOT_DIR}/streamlit.pid"
FRONTEND_LOG_FILE="${ROOT_DIR}/streamlit.log"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"

ensure_venv() {
  if [[ ! -f "${ROOT_DIR}/venv/bin/activate" ]]; then
    echo "Virtual environment not found at venv/. Create it first."
    exit 1
  fi
}

read_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    tr -d '[:space:]' <"$pid_file"
  fi
}

is_pid_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

listening_pid_on_port() {
  local port="$1"
  lsof -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true
}

adopt_or_cleanup_pid_file() {
  local pid_file="$1"
  local port="$2"
  local pid
  pid="$(read_pid_file "$pid_file")"
  if is_pid_running "$pid"; then
    echo "$pid"
    return
  fi

  local port_pid
  port_pid="$(listening_pid_on_port "$port")"
  if [[ -n "$port_pid" ]]; then
    echo "$port_pid" >"$pid_file"
    echo "$port_pid"
    return
  fi

  rm -f "$pid_file"
  echo ""
}

start_backend() {
  ensure_venv
  local pid
  pid="$(adopt_or_cleanup_pid_file "$BACKEND_PID_FILE" "$BACKEND_PORT")"
  if is_pid_running "$pid"; then
    echo "Backend already running (PID: $pid) at http://${BACKEND_HOST}:${BACKEND_PORT}"
    return
  fi

  # shellcheck disable=SC1091
  source "${ROOT_DIR}/venv/bin/activate"
  nohup env PYTHONPATH=backend uvicorn backend.app.main:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" >"${BACKEND_LOG_FILE}" 2>&1 &
  echo $! >"${BACKEND_PID_FILE}"
  sleep 1
  echo "Backend started (PID: $(read_pid_file "$BACKEND_PID_FILE")) at http://${BACKEND_HOST}:${BACKEND_PORT}"
}

start_frontend() {
  ensure_venv
  local pid
  pid="$(adopt_or_cleanup_pid_file "$FRONTEND_PID_FILE" "$FRONTEND_PORT")"
  if is_pid_running "$pid"; then
    echo "Frontend already running (PID: $pid) at http://${FRONTEND_HOST}:${FRONTEND_PORT}"
    return
  fi

  # shellcheck disable=SC1091
  source "${ROOT_DIR}/venv/bin/activate"
  nohup streamlit run src/app.py --server.address "${FRONTEND_HOST}" --server.port "${FRONTEND_PORT}" >"${FRONTEND_LOG_FILE}" 2>&1 &
  echo $! >"${FRONTEND_PID_FILE}"
  sleep 1
  echo "Frontend started (PID: $(read_pid_file "$FRONTEND_PID_FILE")) at http://${FRONTEND_HOST}:${FRONTEND_PORT}"
}

stop_pid_file() {
  local name="$1"
  local pid_file="$2"
  local pid
  pid="$(read_pid_file "$pid_file")"

  if ! is_pid_running "$pid"; then
    rm -f "$pid_file"
    echo "${name} is not running."
    return
  fi

  kill "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! is_pid_running "$pid"; then
      break
    fi
    sleep 0.2
  done

  if is_pid_running "$pid"; then
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$pid_file"
  echo "${name} stopped."
}

status_service() {
  local name="$1"
  local pid_file="$2"
  local port="$3"
  local pid
  pid="$(adopt_or_cleanup_pid_file "$pid_file" "$port")"
  if is_pid_running "$pid"; then
    echo "${name}: running (PID: ${pid})"
  else
    echo "${name}: stopped"
  fi
}

logs_service() {
  local target="$1"
  case "$target" in
    backend) tail -f "$BACKEND_LOG_FILE" ;;
    frontend) tail -f "$FRONTEND_LOG_FILE" ;;
    all) tail -f "$BACKEND_LOG_FILE" "$FRONTEND_LOG_FILE" ;;
    *)
      echo "Unknown logs target: $target"
      exit 1
      ;;
  esac
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/manage.sh start [backend|frontend|all]
  ./scripts/manage.sh stop [backend|frontend|all]
  ./scripts/manage.sh restart [backend|frontend|all]
  ./scripts/manage.sh status
  ./scripts/manage.sh logs [backend|frontend|all]

Defaults:
  start     -> all
  stop      -> all
  restart   -> all
  logs      -> backend
EOF
}

ACTION="${1:-start}"
TARGET="${2:-all}"

case "$ACTION" in
  start)
    case "$TARGET" in
      backend) start_backend ;;
      frontend) start_frontend ;;
      all) start_backend; start_frontend ;;
      *) usage; exit 1 ;;
    esac
    ;;
  stop)
    case "$TARGET" in
      backend) stop_pid_file "Backend" "$BACKEND_PID_FILE" ;;
      frontend) stop_pid_file "Frontend" "$FRONTEND_PID_FILE" ;;
      all) stop_pid_file "Frontend" "$FRONTEND_PID_FILE"; stop_pid_file "Backend" "$BACKEND_PID_FILE" ;;
      *) usage; exit 1 ;;
    esac
    ;;
  restart)
    case "$TARGET" in
      backend) stop_pid_file "Backend" "$BACKEND_PID_FILE"; start_backend ;;
      frontend) stop_pid_file "Frontend" "$FRONTEND_PID_FILE"; start_frontend ;;
      all) stop_pid_file "Frontend" "$FRONTEND_PID_FILE"; stop_pid_file "Backend" "$BACKEND_PID_FILE"; start_backend; start_frontend ;;
      *) usage; exit 1 ;;
    esac
    ;;
  status)
    status_service "Backend" "$BACKEND_PID_FILE" "$BACKEND_PORT"
    status_service "Frontend" "$FRONTEND_PID_FILE" "$FRONTEND_PORT"
    ;;
  logs)
    logs_service "${TARGET:-backend}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
