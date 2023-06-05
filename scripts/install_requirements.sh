#!/bin/bash
#
# Установка виртуального окружения и зависимостей Python.

python3 -m venv .venv
. .venv/bin/activate && pip install -r requirements.txt