# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# 1. Загрузите данные из файла abalone.csv. Это датасет, в котором тре-
# буется предсказать возраст ракушки (число колец) по физическим
# измерениям.

data = pd.read_csv('abalone.csv')

# 2. Преобразуйте признак Sex в числовой: значение F должно перейти
# в -1, I — в 0, M — в 1.

sex_map = {'M': 1, 'F': -1, 'I': 0}
data.Sex = data.Sex.map(sex_map)

# 3. Разделите содержимое файлов на признаки и целевую переменную.
# В последнем столбце записана целевая переменная, в остальных —
# признаки.

X = data.ix[:, :-1]
y = data.ix[:, -1]

# 4. Обучите случайный лес (sklearn.ensemble.RandomForestRegressor)
# с различным числом деревьев: от 1 до 50 (random_state=1). Для
# каждого из вариантов оцените качество работы полученного леса
# на кросс-валидации по 5 блокам. Используйте параметры
# "random_state=1"и "shuffle=True"при создании генератора кросс-
# валидации sklearn.cross_validation.KFold. В качестве меры качества
# воспользуйтесь коэффициентом детерминации (sklearn.metrics.r2_score).

kf = KFold(n_splits=5, shuffle=True, random_state=1)

trigger = False
for n in range(1, 51):
    forest = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    scores = np.array([])
    for train_index, test_index in kf.split(X):
        forest.fit(X.loc[train_index], y.loc[train_index])
        predictions = forest.predict(X.loc[test_index])
        scores = np.append(scores, r2_score(y.loc[test_index], predictions))
    current_score = scores.mean()
    print current_score
    # 5. Определите, при каком минимальном количестве деревьев случай-
    # ный лес показывает качество на кросс-валидации выше 0.52. Это
    # количество и будет ответом на задание.
    if current_score > 0.52 and not trigger:
        print 'n = ', n
        f = open('5-1.txt', 'w')
        f.write(str(n))
        f.close()
        trigger = True

    # 6. Обратите внимание на изменение качества по мере роста числа де-
    # ревьев. Ухудшается ли оно?