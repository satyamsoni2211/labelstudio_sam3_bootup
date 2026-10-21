#!/usr/bin/env bash
# start|stop|restart|status  the native SAM3 backend (background, log in logs/backend.log, pid in logs/backend.pid)
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
pid() { [ -f logs/backend.pid ] && kill -0 "$(cat logs/backend.pid)" 2>/dev/null && cat logs/backend.pid; }
case "$1" in
  start)
    if pid >/dev/null; then echo "already running, pid $(pid)"; exit 0; fi
    cd sam3_backend && ML_BACKEND_PORT=9090 PYTORCH_ENABLE_MPS_FALLBACK=1 nohup .venv/bin/python _wsgi.py > ../logs/backend.log 2>&1 &
    echo $! > "$OLDPWD/logs/backend.pid"; echo "started pid $!, log: logs/backend.log (first start downloads/loads weights, ~1-3 min)";;
  stop) p=$(pid); [ -n "$p" ] && kill "$p"
        for q in $(lsof -tiTCP:9090 -sTCP:LISTEN 2>/dev/null); do kill "$q"; done   # the python child, if the wrapper is gone
        [ -n "$p" ] && echo stopped || echo "not running";;
  restart) "$0" stop; sleep 2; "$0" start;;
  status) pid >/dev/null && echo "running pid $(pid)" || echo "not running"; curl -s localhost:9090/health; echo;;
esac
