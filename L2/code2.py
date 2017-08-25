# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
import sklearn.datasets

# 1. Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект, у которого
# признаки записаны в поле data, а целевой вектор — в поле target.

boston = sklearn.datasets.load_boston()

# 2. Приведите признаки в выборке к одному масштабу при помощи
# функции sklearn.preprocessing.scale.

boston.data = scale(boston.data)

# 3. Переберите разные варианты параметра метрики p по сетке от 1 до
# 10 с таким шагом, чтобы всего было протестировано 200 вариантов.
# Используйте KNeighborsRegressor с n_neighbors=5 и
# weights=’distance’ — данный параметр добавляет
# в алгоритм веса, зависящие от расстояния до ближайших соседей. В
# качестве метрики качества используйте среднеквадратичную ошиб-
# ку. Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации
# по 5 блокам с random_state = 42, не забудьте включить перемешивание выборки.

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy = {}
for p in np.linspace(1, 10, num=200):
	knr = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
	cvs = cross_val_score(knr, boston.data, boston.target, cv=kf)
	accuracy.update({ p: cvs.mean() })

# 4. Определите, при каком p качество на кросс-валидации оказалось
# оптимальным. Обратите внимание, что cross_val_score возвращает
# массив показателей качества по блокам; необходимо максимизи-
# ровать среднее этих показателей. Это значение параметра и будет
# ответом на задачу.

best_accuracy = max(accuracy.values())
best_p = accuracy.keys()[accuracy.values().index(best_accuracy)]

print best_p, best_accuracy
f = open('2-2.txt', 'w')
f.write(str(best_p))
f.close()