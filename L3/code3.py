# -*- coding: utf-8 -*-

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

def answer(filename='3-3.txt', *answer):
    file = open(filename, 'w')
    ans = ''
    for i in answer:
        ans += str(i) + ' '
    ans = ans[:-1]
    file.write(ans)
    file.close()

def show(*names):
    for n in names:
        print n + ': ' + str(eval(n))

# 1. Загрузите файл classification.csv. В нем записаны истинные классы
# объектов выборки (колонка true) и ответы некоторого классифика-
# тора (колонка predicted).

data = pd.read_csv('classification.csv')
print data.keys()

# 2. Заполните таблицу ошибок классификации.

TP = data[(data.pred == 1) & (data.true == 1)].count().get(0)
TN = data[(data.pred == 0) & (data.true == 0)].count().get(0)
FP = data[(data.pred == 1) & (data.true == 0)].count().get(0)
FN = data[(data.pred == 0) & (data.true == 1)].count().get(0)
show('TP', 'FP', 'FN', 'TN')
answer('3-3-1.txt', TP, FP, FN, TN)

# 3. Посчитайте основные метрики качества классификатора.

accuracy = accuracy_score(data.true, data.pred)
precision = precision_score(data.true, data.pred)
recall = recall_score(data.true, data.pred)
f1 = f1_score(data.true, data.pred)
show('accuracy', 'precision', 'recall', 'f1')
answer('3-3-2.txt', accuracy, precision, recall, f1)

# 4. Имеется четыре обученных классификатора. В файле scores.csv за-
# писаны истинные классы и значения степени принадлежности по-
# ложительному классу для каждого классификатора на некоторой выборке:
# • для логистической регрессии — вероятность положительного класса (колонка score_logreg),
# • для SVM — отступ от разделяющей поверхности (колонка score_svm),
# • для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
# • для решающего дерева — доля положительных объектов в листе (колонка score_tree).
# Загрузите этот файл.

scores = pd.read_csv('scores.csv')
print scores.keys()

# 5. Посчитайте площадь под ROC-кривой для каждого классифика-
# тора. Какой классификатор имеет наибольшее значение метрики
# AUC-ROC?

auc = {score: roc_auc_score(scores.true, scores[score]) for score in scores.keys()[1:]}
m = max(auc, key=auc.get)
print m
answer('3-3-3.txt', m)

# 6. Какой классификатор достигает наибольшей точности (Precision)
# при полноте (Recall) не менее 70% (укажите название столбца с от-
# ветами этого классификатора)? Какое значение точности при этом
# получается?

curve = {}
for score in scores.keys()[1:]:
    df = DataFrame(columns=('precision', 'recall'))
    df.precision, df.recall, thresholds = precision_recall_curve(scores.true, scores[score])
    curve.update({score: df[df['recall'] >= 0.7]['precision'].max()})
print curve
best_model = max(curve, key=curve.get)
print best_model
answer('3-3-4.txt', best_model)