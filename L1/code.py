# -*- coding: utf-8 -*-

import numpy as np
import pandas
import inspect
from collections import Counter
from sklearn.tree import DecisionTreeClassifier as dtc


data = pandas.read_csv('titanic.csv')
data.set_index('PassengerId')

print data.keys()

def answer(*answer):
    filename = inspect.stack()[1][3] + '.txt'
    file = open(filename, 'w')
    ans = ''
    for i in answer:
        ans += str(i) + ' '
    ans = ans[:-1]
    file.write(ans)

# 1. Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите
# два числа через пробел.

def task1():
    m = data['Sex'].value_counts()['male']
    f = data['Sex'].value_counts()['female']
    print 'male: ', m
    print 'female: ', f
    answer(m, f)

# 2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен),
# округлив до двух знаков.

def task2():
    ans = round(float(data.Survived.value_counts()[1]) / data.Survived.count() * 100, 2)
    print 'Survived: ', ans, '%'
    answer(ans)

# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен),
# округлив до двух знаков.

def task3():
    ans = round(float(data.Pclass.value_counts()[1]) / data.Pclass.count() * 100, 2)
    print '1st class: ', ans, '%'
    answer(ans)

# 4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.

def task4():
    mean = data.Age.mean()
    median = data.Age.median()
    print 'Mean age: ', mean
    print 'Median age: ', median
    answer(mean, median)

# 5. Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.

def task5():
    ans = round(np.corrcoef(data['SibSp'], data['Parch'])[0][-1], 2)
    print 'Correlation: ', ans
    answer(ans)

# 6. Какое самое популярное женское имя на корабле?

def task6():
    def get_first_name(name):
        f = name.find('.')
        name = name[f+1:]
        name = name.strip()
        if name[-1] == ')':
            name = name[name.find('(')+1:-1]
        return name.split()

    names = []
    for i, row in data.iterrows():
        if row['Sex'] == 'female':
            names.extend(get_first_name(row['Name']))
    ans = Counter(names).most_common()[0][0]
    print ans
    answer(ans)

# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare),
# возраст пассажира (Age) и его пол (Sex).
# Обратите внимание, что признак Sex имеет строковые значения.
# Выделите целевую переменную — она записана в столбце Survived.
# В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст.
# Такие записи при чтении их в pandas принимают значение nan.
# Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию
# (речь идет о параметрах конструктора DecisionTreeСlassifier).
# Вычислите важности признаков и найдите два признака с наибольшей важностью.
# Их названия будут ответами для данной задачи (в качестве ответа укажите
# названия признаков через запятую или пробел, порядок не важен).

def lesson2():
    N_OF_IMPORTANT = 2

    btree_data = data.loc[~np.isnan(data['Age']), ['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
    btree_data.Sex = btree_data.Sex.map({'male': 1, 'female': 0})
    tree = dtc(random_state=241)
    tree.fit(btree_data.loc[:, :'Sex'], btree_data.loc[:, 'Survived'])
    importances = tree.feature_importances_
    indices = np.argsort(importances)[::-1]
    columns = list(btree_data.columns.values)
    ans = [columns[indices[i]] for i in range(N_OF_IMPORTANT)]
    answer(ans)


task1()
task2()
task3()
task4()
task5()
task6()
lesson2()
