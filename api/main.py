
from fastapi import Depends, FastAPI, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .config import set_looger
from .utils import load_model, make_prediction

app = FastAPI()
logger = set_looger()
classifier = load_model()


class Features(BaseModel):
    """
    Модель набора факторных признаков, используемых для предсказаний.

    1.   RevolvingUtilizationOfUnsecuredLines  -  float64
    2.   age                                   -  float64
    3.   NumberOfTime30-59DaysPastDueNotWorse  -  int64
    4.   DebtRatio                             -  float64
    5.   MonthlyIncome                         -  float64
    6.   NumberOfOpenCreditLinesAndLoans       -  int64
    7.   NumberOfTimes90DaysLate               -  int64
    8.   NumberOfTime60-89DaysPastDueNotWorse  -  int64
    9.   NumberOfDependents                    -  float64
    10.  RealEstateLoansOrLines                -  object
    11.  GroupAge                              -  object
    """

    revolving_utilization_of_unsecured_lines: float = Field(
        Query(alias="RevolvingUtilizationOfUnsecuredLines")
    )
    age: float = Field(Query(alias="age"))
    number_of_time_30_59_days_past_due_not_worse: int = Field(
        Query(alias="NumberOfTime30-59DaysPastDueNotWorse")
    )
    debt_ratio: float = Field(Query(alias="DebtRatio"))
    monthly_income: float = Field(Query(alias="MonthlyIncome"))
    number_of_open_credit_lines_and_loans: int = Field(
        Query(alias="NumberOfOpenCreditLinesAndLoans")
    )
    number_of_times_90_days_late: int = Field(Query(alias="NumberOfTimes90DaysLate"))
    number_of_time_60_89_days_past_due_not_worse: int = Field(
        Query(alias="NumberOfTime60-89DaysPastDueNotWorse")
    )
    number_of_dependents: float = Field(Query(alias="NumberOfDependents"))
    real_estate_loans_or_lines: str = Field(Query(alias="RealEstateLoansOrLines"))
    group_age: str = Field(Query(alias="GroupAge"))


@app.get("/", include_in_schema=False)
def root():
    """
    Корневая директория сервиса.
    """
    return RedirectResponse("/docs")


@app.get("/predict/", summary="Получить предсказание и вероятность")
def predict(features: Features = Depends()):
    """
    Отправить факторные признаки и получить предсказание с вероятностью.

    Факторные признаки:
    - RevolvingUtilizationOfUnsecuredLines: Общий баланс средств.
    - age: Возраст заемщика
    - NumberOfTime30-59DaysPastDueNotWorse: Сколько раз за последние 2 года наблюдалась просрочка 30-59 дней.
    - DebtRatio: Ежемесячные расходы (платеж по долгам, алиментам, расходы на проживания) деленные на месячный доход.
    - MonthlyIncome: Ежемесячный доход.
    - NumberOfOpenCreditLinesAndLoans: Количество открытых кредитов (напрмер, автокредит или ипотека) и кредитных карт.
    - NumberOfTimes90DaysLate: Сколько раз наблюдалась просрочка (90 и более дней).
    - NumberRealEstateLoansOrLines: Количество кредиов (в том числе под залог жилья)
    - NumberOfTime60-89DaysPastDueNotWorse: Сколько раз за последние 2 года заемщик задержал платеж на 60-89 дней.
    - NumberOfDependents: Количество иждивенцев на попечении (супруги, дети и др).
    - RealEstateLoansOrLines: Закодированное количество кредиов (в том числе под залог жилья) - чем больше код буквы, тем больше кредитов
    - GroupAge: закодированная возрастная группа - чем больше код, тем больше возраст.
    """
    data = {
        "RevolvingUtilizationOfUnsecuredLines": features.revolving_utilization_of_unsecured_lines,
        "age": features.age,
        "NumberOfTime30-59DaysPastDueNotWorse": features.number_of_time_30_59_days_past_due_not_worse,
        "DebtRatio": features.debt_ratio,
        "MonthlyIncome": features.monthly_income,
        "NumberOfOpenCreditLinesAndLoans": features.number_of_open_credit_lines_and_loans,
        "NumberOfTimes90DaysLate": features.number_of_times_90_days_late,
        "NumberOfTime60-89DaysPastDueNotWorse": features.number_of_time_60_89_days_past_due_not_worse,
        "NumberOfDependents": features.number_of_dependents,
        "RealEstateLoansOrLines": features.real_estate_loans_or_lines,
        "GroupAge": features.group_age,
    }
    logger.info(f"Получен запрос со следующими параметрами:\n{data}")
    prediction, probability = make_prediction(data, classifier)
    logger.info(
        f"Предсказанный класс: {prediction}, вероятность класса 1: {probability}"
    )
    return {
        "msg": "Получены предсказание класса и вероятность класса 1",
        "result": {"prediction": prediction, "probability": probability},
    }
