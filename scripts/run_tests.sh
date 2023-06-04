#!/bin/bash
#
# Запуск тестов pytest.

project_dir="$(dirname "$(dirname "$(readlink -f "$0")")")"
pytest="$project_dir/.venv/bin/pytest"
$pytest