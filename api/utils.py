import os
import time

import pandas as pd
from catboost import CatBoostClassifier, Pool

from .config import PROJECT_DIR


def make_prediction(data: dict, classifier: CatBoostClassifier):
    """
    Вернуть предсказанный класс и вероятность класса 1.
    """
    data = pd.DataFrame.from_dict([data])
    pool = Pool(data, cat_features=["RealEstateLoansOrLines", "GroupAge"])
    prediction = str(classifier.predict(pool)[0])
    probability = str(classifier.predict_proba(pool)[0, 1])
    return prediction, probability

def load_model():
    """
    Загрузить модель машинного обучения,
    заранее обученную в другом процессе.
    """
    classifier = CatBoostClassifier()
    models_dir = os.path.join(PROJECT_DIR, "models")
    model_path = os.path.join(models_dir, "catboost_model.json")
    while not os.path.exists(model_path):
        time.sleep(10)
    classifier.load_model(model_path, format="json")
    return classifier