# -*- coding: utf-8 -*-

import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.base import Bunch

# 1. Загрузите обучающую и тестовую выборки из файлов perceptron-
# train.csv и perceptron-test.csv. Целевая переменная записана в первом
# столбце, признаки — во втором и третьем.

data = pandas.read_csv('perceptron-train.csv', header=None)
train, test = Bunch(), Bunch()
train.data, train.target = data.loc[:, 1:], data.loc[:, 0]
data = pandas.read_csv('perceptron-test.csv', header=None)
test.data, test.target = data.loc[:, 1:], data.loc[:, 0]

# 2. Обучите персептрон со стандартными параметрами и random_state=241

perc = Perceptron(random_state=241)
perc.fit(train.data, train.target) # learning

# 3. Подсчитайте качество (долю правильно классифицированных объ-
# ектов, accuracy) полученного классификатора на тестовой выборке.

accuracy = perc.score(test.data, test.target) # predicting
print accuracy

# 4. Нормализуйте обучающую и тестовую выборку с помощью класса
# StandardScaler.

scaler = StandardScaler() # scaling
train_scaled, test_scaled = Bunch(), Bunch()
train_scaled.data = scaler.fit_transform(train.data)
test_scaled.data = scaler.transform(test.data)
train_scaled.target, test_scaled.target = train.target, test.target

# 5. Обучите персептрон на новых выборках. Найдите долю правиль-
# ных ответов на тестовой выборке.

perc.fit(train_scaled.data, train_scaled.target)
accuracy_scaled = perc.score(test_scaled.data, test_scaled.target)
print accuracy_scaled

# 6. Найдите разность между качеством на тестовой выборке после нор-
# мализации и качеством до нее. Это число и будет ответом на зада-
# ние.

difference = abs(accuracy_scaled - accuracy)
print difference

f = open('3.txt', 'w')
f.write(str(difference))
f.close()