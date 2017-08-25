# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

# 1. Загрузите объекты из новостного датасета 20 newsgroups, относя-
# щиеся к категориям "космос"и "атеизм".

ng = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

# 2. Вычислите TF-IDF-признаки для всех текстов. Обратите внима-
# ние, что в этом задании мы предлагаем вам вычислить TF-IDF по
# всем данным.

vectorizer = TfidfVectorizer()
print type(ng.data)
X = vectorizer.fit_transform(ng.data) # count tf-idf
print X
y = ng.target
feature_map = vectorizer.get_feature_names() # look for words with ith feature

# 3. Подберите минимальный лучший параметр C из множества
# [10^−5, 10^−4, ... 10^4, 10^5] для SVM с линейным ядром  при помощи кросс-
# валидации по 5 блокам. Укажите параметр random_state=241 и
# для SVM, и для KFold. В качестве меры качества используйте до-
# лю верных ответов (accuracy).

C_grid = {'C': np.power(10.0, np.arange(-5, 6))}
kf = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, C_grid, scoring='accuracy', cv=kf, n_jobs=-1)
gs.fit(X, y)

# 4. Обучите SVM по всей выборке с лучшим параметром C, найденным
# на предыдущем шаге.

clf = gs.best_estimator_
clf.fit(X, y)

# 5. Найдите 10 слов с наибольшим по модулю весом. Они являются
# ответом на это задание. Укажите их через запятую, в нижнем ре-
# гистре, в лексикографическом порядке.

weights = np.absolute(clf.coef_.toarray())
max_weights = sorted(zip(weights[0], feature_map))[-10:]
max_weights.sort(key=lambda x: x[1])
print max_weights

f = open('3-1-2.txt', 'w')
for w, c in max_weights[:-1]:
	f.write(c)
	f.write(',')
f.write(max_weights[-1][1])
f.close()