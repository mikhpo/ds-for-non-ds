import logging
import os
from logging import handlers
from pathlib import Path

from catboost import CatBoostClassifier

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(APP_DIR).resolve().parent

from .config import PROJECT_DIR


def load_model():
    """
    Загрузить модель машинного обучения,
    заранее обученную в другом процессе.
    """
    classifier = CatBoostClassifier()
    models_dir = os.path.join(PROJECT_DIR, "models")
    model_path = os.path.join(models_dir, "catboost_model.json")
    classifier.load_model(model_path, format="json")
    return classifier


def set_looger():
    """
    Установка настроек логирования.
    """
    # Определить путь сохранения логов.
    app_name = Path(APP_DIR).resolve().stem
    logs_dir = os.path.join(PROJECT_DIR, "logs", app_name)
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{app_name}.log")

    # Общие настройки логирования.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Логировать в консоль.
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    # Логировать в файл.
    formatter = logging.Formatter(
        fmt="[{asctime}] [{levelname}] [{filename} -> {funcName} -> {lineno}] {message}",
        datefmt="%d.%m.%Y %H:%M:%S",
        style="{",
    )
    timed_rotating_handler = handlers.TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=7
    )
    timed_rotating_handler.setFormatter(formatter)
    logger.addHandler(timed_rotating_handler)
    return logger
