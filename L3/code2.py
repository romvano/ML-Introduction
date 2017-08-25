# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as rms
from sklearn.metrics import roc_auc_score
from math import exp

# Загрузите данные из файла data-logistic.csv. Это двумерная выбор-
# ка, целевая переменная на которой принимает значения -1 или 1.

data = pd.read_csv('data-logistic.csv', header=None)
x = data.loc[:, 1:]
y = data[0]

print x.keys()

K = 0.1
ERROR = 1e-5

def sigma_y(i, w1, w2):
    return 1. / (1. + exp(-y[i] * (w1*x[1][i] + w2*x[2][i])))

def delta_for_w(w_index, w1, w2, C):
    addition = sum((
        y[i] * x[w_index][i] * (1. - sigma_y(i, w1, w2)) for i in np.arange(0, len(y))
    ))
    addition *= K / len(y)
    addition -= K * C * (w1 if w_index == 1 else w2)
    return addition

# Реализуйте градиентный спуск для обычной и L2-регуляризованной
# (с коэффициентом регуляризации 10) логистической регрессии. Ис-
# пользуйте длину шага k=0.1. В качестве начального приближения
# используйте вектор (0, 0).

def gradient_regressor(C, iterations_remaining=10000):
    changed_w1, changed_w2 = 0., 0.
    while iterations_remaining:
        iterations_remaining -= 1
        w1, w2 = changed_w1, changed_w2
        changed_w1 = w1 + delta_for_w(1, w1, w2, C)
        changed_w2 = w2 + delta_for_w(2, w1, w2, C)
        if np.sqrt(rms([w1, w2], [changed_w1, changed_w2])) <= ERROR:
            break
    return changed_w1, changed_w2

def sigma(xi, w1, w2):
    return 1. / (1 + np.exp(-w1 * xi[1] - w2 * xi[2]))

# Запустите градиентный спуск и доведите до сходимости (евклидово
# расстояние между векторами весов на соседних итерациях долж-
# но быть не больше 1e-5). Рекомендуется ограничить сверху число
# итераций десятью тысячами.

w1, w2 = gradient_regressor(0.)
l2w1, l2w2 = gradient_regressor(10.)

print w1, w2, l2w1, l2w2

scores = x.apply(lambda xi: sigma(xi, w1, w2), axis=1)
l2scores = x.apply(lambda xi: sigma(xi, l2w1, l2w2), axis=1)

# Какое значение принимает AUC-ROC на обучении без регуляри-
# зации и при ее использовании?

auc_score = roc_auc_score(y, scores)
l2_auc_score = roc_auc_score(y, l2scores)

print auc_score
print l2_auc_score

f = open('3-2.txt', 'w')
f.write(str(auc_score) + ' ' + str(l2_auc_score))
f.close()