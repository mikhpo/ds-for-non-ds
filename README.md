# DS-for-non-DS
Материалы для прохождения учебного курса "DS for non DS" (Введение в Data Science, практический курс (16 мая - 6 июня)).

## Содержание репозитория
* `notebooks` - ноутбуки Jupyter
* `models` - сохраненные модели

## Google Colab
Ноутбук доступен в Google Colab по адресу https://colab.research.google.com/drive/1RJVk4It2gAIzuc7Zeru_P8AlamfJ_lt7

## Описание задачи

### Кредитный скоринг

Ваша задача состоит в том, чтобы по различным характеристикам клиентов спрогнозировать целевую переменную - имел клиент просрочку 90 и более дней или нет (и если имел, то банк не будет выдавать кредит этому клиенту, а иначе будет). 

Ниже находится описание признаков клиентов.

Целевая переменная
**SeriousDlqin2yrs**: Клиент имел просрочку 90 и более дней
- **RevolvingUtilizationOfUnsecuredLines**: Общий баланс средств (total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits).
- **age**: Возраст заемщика
- **NumberOfTime30**-59DaysPastDueNotWorse: Сколько раз за последние 2 года наблюдалась просрочка 30-59 дней.
- **DebtRatio**: Ежемесячные расходы (платеж по долгам, алиментам, расходы на проживания) деленные на месячный доход.
- **MonthlyIncome**: Ежемесячный доход.
- **NumberOfOpenCreditLinesAndLoans**: Количество открытых кредитов (напрмер, автокредит или ипотека) и кредитных карт.
- **NumberOfTimes90DaysLate**: Сколько раз наблюдалась просрочка (90 и более дней).
- **NumberRealEstateLoansOrLines**: Количество кредиов (в том числе под залог жилья)
- **RealEstateLoansOrLines**: Закодированное количество кредиов (в том числе под залог жилья) - чем больше код буквы, тем больше кредитов
- **NumberOfTime60**-89DaysPastDueNotWorse: Сколько раз за последние 2 года заемщик задержал платеж на 60-89 дней.
- **NumberOfDependents**: Количество иждивенцев на попечении (супруги, дети и др).
- **GroupAge**: закодированная возрастная группа - чем больше код, тем больше возраст.

Таблица находится в схеме `public` под названием `credit_scoring`. 
