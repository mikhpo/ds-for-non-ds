import os
from http import HTTPStatus
from typing import Union

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ..config import PROJECT_DIR
from ..main import app

client = TestClient(app)

dtypes = Union[float, int, str]


@pytest.fixture
def data():
    """
    Получить набор исходных данных для тестирования в формате словаря.
    """
    data_dir = os.path.join(PROJECT_DIR, "data")
    data_file = os.path.join(data_dir, "credit_scoring.csv")
    data_df = pd.read_csv(data_file)
    data_df.dropna(inplace=True)
    data_row = data_df.iloc[0]
    data_dict = data_row.to_dict()
    return data_dict


def test_root():
    """
    Тест получения кода ответа OK на запрос корневого маршрута сервиса.
    """
    url = "/"

    response = client.get(url)
    assert response.status_code == HTTPStatus.OK

    response = client.get(url, follow_redirects=False)
    assert response.is_redirect


def test_docs():
    """
    Тест доступности страницы документации API, которая генерируется автоматически.
    """
    url = "/docs/"
    response = client.get(url)
    assert response.status_code == HTTPStatus.OK


def test_predict(data: dict[str, dtypes]):
    """
    Тест страницы получения предсказания.
    """
    url = "/predict/"

    response = client.get(url, params={})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    response = client.get(url, params=data)
    assert response.status_code == HTTPStatus.OK

    response_dict: dict = response.json()
    response_values = response_dict.values()
    assert "Получены предсказание класса и вероятность класса 1" in response_values
