# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import log_loss
from matplotlib import pyplot as plt
import operator, os
from sklearn.ensemble import RandomForestClassifier as RFC

def answer(filename, *args):
    f = open(filename, 'w')
    s = ' '.join([str(a) for a in args])
    f.write(s)
    f.close()

def sigma(y):
    return 1 / (1 + np.exp(-y))

def plot(train, test, name_postfix):
    plt.figure()
    plt.plot(train, 'r', linewidth=3)
    plt.plot(test, 'b', linewidth=3)
    plt.legend(['train', 'test'])
    plt.savefig('plots/rate_' + str(name_postfix) + '.png')

# 1. Загрузите выборку из файла gbm-data.csv с помощью pandas и пре-
# образуйте ее в массив numpy (параметр values у датафрейма). В
# 3первой колонке файла с данными записано, была или нет реакция.
# Все остальные колонки (d1 - d1776) содержат различные характери-
# стики молекулы, такие как размер, форма и т.д. Разбейте выборку
# на обучающую и тестовую, используя функцию train_test_split с
# параметрами test_size = 0.8 и random_state = 241.

data = pd.read_csv('gbm-data.csv')

X = data.ix[:, 1:].values
y = data.ix[:, 0].values

print X.size, y.size
PLOT_DIR = 'plots'
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# 2. Обучите GradientBoostingClassifier с параметрами n_estimators=250,
# verbose=True, random_state=241 и для каждого значения learning_rate
# из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
for lr in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = GBC(n_estimators=250, learning_rate=lr, verbose=True)
    clf.fit(X_train, y_train)
    # • Используйте метод staged_decision_function для предсказа-
    # ния качества на обучающей и тестовой выборке на каждой итерации.
    # • Преобразуйте полученное предсказание по формуле
    # где y_pred — предсказанное значение.
    y_pred_train = [sigma(sdf) for sdf in clf.staged_decision_function(X_train)]
    y_pred_test = [sigma(sdf) for sdf in clf.staged_decision_function(X_test)]

    # • Вычислите и постройте график значений log-loss на обучаю-
    # щей и тестовой выборках, а также найдите минимальное зна-
    # чение метрики и номер итерации, на которой оно достигается.
    train_loss = np.array([log_loss(y_train, y_pred) for y_pred in y_pred_train]) # count log_loss
    test_loss = np.array([log_loss(y_test, y_pred) for y_pred in y_pred_test])
    plot(train_loss, test_loss, lr)
    min_loss_index, min_loss = min(enumerate(test_loss), key=operator.itemgetter(1))
    # 4. Приведите минимальное значение log-loss на тестовой выборке и
    # номер итерации, на котором оно достигается, при learning_rate = 0.2.
    if lr == 0.2:
        answer('5-2-2.txt', min_loss, min_loss_index)
        # 3. Как можно охарактеризовать график качества на тестовой выборке,
        # начиная с некоторой итерации: переобучение (overfitting) или
        # недообучение (underfitting)?
        fitting = 'overfitting' if test_loss[int(3.*test_loss.size/4.) :].mean() > test_loss.mean() else 'underfitting'
        answer('5-2-1.txt', fitting)

# 5. На этих же данных обучите RandomForestClassifier с количеством
# деревьев, равным количеству итераций, на котором достигается
# наилучшее качество у градиентного бустинга из предыдущего пункта,
# random_state=241 и остальными параметрами по умолчанию.
# Какое значение log-loss на тесте получается у этого случайного леса?

forest = RFC(n_estimators=min_loss_index, random_state=241)
forest.fit(X_train, y_train)
y_forest = forest.predict_proba(X_test)
forest_loss = log_loss(y_test, y_forest)
print forest_loss
answer('5-2-3.txt', forest_loss)