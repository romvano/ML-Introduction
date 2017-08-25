# -*- coding: utf-8 -*-

import pandas
from sklearn.svm import SVC

# 1. Загрузите выборку из файла svm-data.csv. В нем записана двумер-
# ная выборка (целевая переменная указана в первом столбце, при-
# знаки — во втором и третьем).

data = pandas.read_csv('svm-data.csv', header=None)
X = data.loc[:, 1:]
y = data.loc[:, 0]

# 2. Обучите классификатор с линейным ядром, параметром C = 100000
# и random_state=241.

svc = SVC(kernel='linear', C=100000, random_state=241)

svc.fit(X, y)
print svc.support_

for a in svc.__dict__:
    print a, '\t', svc.__dict__[a]

f = open('3-1.txt', 'w')

# 3. Найдите номера объектов, которые являются опорными (нумера-
# ция с единицы). Они будут являться ответом на задание. Обратите
# внимание, что в качестве ответа нужно привести номера объектов
# в возрастающем порядке через запятую. Нумерация начинается с 1.

for n in svc.support_:
    f.write(str(n + 1).encode('ascii'))
    f.write(',')
f.close()