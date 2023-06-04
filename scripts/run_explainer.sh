#!/bin/bash
#
# Запуск дашборда ExplainerDashboard.

project_dir="$(dirname "$(dirname "$(readlink -f "$0")")")"
python="$project_dir/.venv/bin/python"
explainer="$project_dir/explainer/main.py"
$python $explainer