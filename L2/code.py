# -*- coding: utf-8 -*-

import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

# Загрузите выборку Wine
data = pandas.read_csv('wine.data.csv', header=None)

# Извлеките из данных признаки и классы. Класс записан в первом столбце (три варианта),
# признаки — в столбцах со второго по последний. Более подробно о сути признаков можно
# прочитать по адресу https://archive.ics.uci.edu/ml/datasets/Win

# Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold).
# Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True).
# Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром
# random_state=42. В качестве меры качества используйте долю верных ответов (accuracy).

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Найдите точность классификации на кросс-валидации для метода k ближайших соседей
# (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50. При каком k получилось оптимальное
# качество? Чему оно равно (число в интервале от 0 до 1)?
# Данные результаты и будут ответами на вопросы 1 и 2.

k_neighbors_accuracy = {}
for n in range(1, 51):
    nc = KNeighborsClassifier(n_neighbors=n)
    cvs = cross_val_score(nc, data.loc[:, 1:], data.loc[:, 0], cv=kf)
    k_neighbors_accuracy.update({ n: cvs.mean() })

best_accuracy = max(k_neighbors_accuracy.values())
best_n = k_neighbors_accuracy.keys()[ k_neighbors_accuracy.values().index(best_accuracy)]
print best_n, best_accuracy

# Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale.
# Снова найдите оптимальное k на кросс-валидации.

scaled_data = scale(data.loc[:, 1:])
k_neighbors_accuracy_scaled = {}
for n in range(1, 51):
	nc = KNeighborsClassifier(n_neighbors=n)
	cvs_scaled = cross_val_score(nc, scaled_data, data.loc[:, 0], cv=kf)
	k_neighbors_accuracy_scaled.update({ n: cvs_scaled.mean() })

# Какое значение k получилось оптимальным после приведения признаков к одному масштабу?
# Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?

best_accuracy_scaled = max(k_neighbors_accuracy_scaled.values())
best_n_scaled = k_neighbors_accuracy_scaled.keys()[k_neighbors_accuracy_scaled.values().index(best_accuracy_scaled)]
print best_n_scaled, best_accuracy_scaled

for n in range(1, 5):
	f = open('1-' + str(n) + '.txt', 'w')
	f.write([str(best_n), str(best_accuracy), str(best_n_scaled), str(best_accuracy_scaled)][n-1])
	f.close()