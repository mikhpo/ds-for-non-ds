#!/bin/bash
#
# Запуск последовательно двух сервисов в фоновом режиме.

project_dir="$(dirname "$(dirname "$(readlink -f "$0")")")"
explainer_script="$project_dir/scripts/run_explainer.sh"
api_script="$project_dir/scripts/run_api.sh"
sh $explainer_script &
sh $api_script &
wait