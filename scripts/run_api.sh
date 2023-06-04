#!/bin/bash
#
# Запуск сервиса FastAPI.

project_dir="$(dirname "$(dirname "$(readlink -f "$0")")")"
uvicorn="$project_dir/.venv/bin/uvicorn"
app="api.main:app"
flags="--reload"
$uvicorn $app $flags