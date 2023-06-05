#!/bin/bash
#
# Запуск сервиса FastAPI.

project_dir="$(dirname "$(dirname "$(readlink -f "$0")")")"
uvicorn="$project_dir/.venv/bin/uvicorn"
app="api.main:app --host 0.0.0.0 --port 8002"
flags="--reload"
$uvicorn $app $flags