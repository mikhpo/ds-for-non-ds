import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.model_selection import train_test_split


def construct_filepath():
    """
    Получить адрес корневой директории проекта.
    """
    project_dir = Path(__file__).resolve().parent.parent
    return project_dir


def load_data(project_dir: str):
    """
    Загрузить набор данных и разбить на факторные признаки и целевой признак.
    """
    TARGET_LABEL = "SeriousDlqin2yrs"
    data_dir = os.path.join(project_dir, "data")
    data_set = os.path.join(data_dir, "credit_scoring.csv")
    data = pd.read_csv(data_set)
    y = data[TARGET_LABEL]
    x = data.drop(columns=[TARGET_LABEL])
    categorical_features = [
        column for column in x.columns if x[column].dtype == "object"
    ]
    return x, y, categorical_features


def fit_model(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    categorical_features: list[str],
):
    """
    Обучить модель CatBoost.
    """
    train_pool = Pool(x_train, y_train, cat_features=categorical_features)
    test_pool = Pool(x_test, y_test, cat_features=categorical_features)
    PARAMS = {
        "iterations": 200,
        "learning_rate": 0.1,
        "auto_class_weights": "Balanced",
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "logging_level": "Verbose",
        "allow_writing_files": False,
        "use_best_model": True,
    }
    model = CatBoostClassifier(**PARAMS)
    model.fit(train_pool, eval_set=test_pool, verbose=False, plot=False)
    return model


def save_model(model: CatBoostClassifier, project_dir: str):
    """
    Сохранить модель для использования в продуктиве.
    """
    models_dir = os.path.join(project_dir, "models")
    filename = os.path.join(models_dir, "catboost_model.json")
    model.save_model(filename, format="json")


def main():
    """
    Создать и запустить дашборд.
    """
    project_dir = construct_filepath()
    x, y, categorical_features = load_data(project_dir)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = fit_model(x_train, x_test, y_train, y_test, categorical_features)
    save_model(model, project_dir)
    explainer = ClassifierExplainer(model, x_test, y_test)
    dashboard = ExplainerDashboard(explainer)
    dashboard.run()


if __name__ == "__main__":
    main()
