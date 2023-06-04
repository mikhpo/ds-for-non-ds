import pandas as pd
from catboost import CatBoostClassifier, Pool


def make_prediction(data: dict, classifier: CatBoostClassifier):
    """
    Вернуть предсказанный класс и вероятность класса 1.
    """
    data = pd.DataFrame.from_dict([data])
    pool = Pool(data, cat_features=["RealEstateLoansOrLines", "GroupAge"])
    prediction = str(classifier.predict(pool)[0])
    probability = str(classifier.predict_proba(pool)[0, 1])
    return prediction, probability
