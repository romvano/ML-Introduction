# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def answer(filename, *answer):
    file = open(filename, 'w')
    ans = ''
    for i in answer:
        ans += str(i) + ' '
    ans = ans[:-1]
    file.write(ans)
    file.close()

# 1. Загрузите данные close_prices.csv. В этом файле приведены цены
# акций 30 компаний на закрытии торгов за каждый день периода.

data = pd.read_csv('close_prices.csv')
useful_data = data.ix[:, 1:]

# 2. На загруженных данных обучите преобразование PCA с числом
# компоненты равным 10. Скольких компонент хватит, чтобы объяс-
# нить 90% дисперсии?

pca = PCA(n_components=10)
transformed_data = pca.fit_transform(useful_data) # transform n default features to m new ones

# how many features explain 90% of dispersion?
component_dispersion = pca.explained_variance_ratio_
i = 1
while np.sum(component_dispersion[:i]) < 0.9:
    i += 1
print i
answer('4-2-1.txt', i)

# 3. Примените построенное преобразование к исходным данным и возь-
# мите значения первой компоненты.

first_component = transformed_data[:, 0]

# 4. Загрузите информацию об индексе Доу-Джонса из файла djia_prices.csv.
# Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?

dj = pd.read_csv('djia_index.csv')
corr = np.corrcoef(first_component, dj.ix[:, 1])[0, 1]
print corr
answer('4-2-2.txt', corr)

# 5. Какая компания имеет наибольший вес в первой компоненте?

company = useful_data.keys()[pca.components_[0].argmax()]
print company
answer('4-2-3.txt', company)