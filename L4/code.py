# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

def transform_data(data):
    # present data correctly
    data.FullDescription = data.FullDescription.replace('[^a-zA-Z0-9]', ' ', regex=True).str.lower()
    return data

# 1. Загрузите данные об описаниях вакансий и соответствующих годо-
# вых зарплатах из файла salary-train.csv.
# 2. Проведите предобработку:
# • Приведите тексты к нижнему регистру.
# • Замените все, кроме букв и цифр, на пробелы — это облегчит
# дальнейшее разделение текста на слова.

data = transform_data(pd.read_csv('salary-train.csv'))
test = transform_data(pd.read_csv('salary-test-mini.csv'))

cols = data.keys()
print data.keys()
print data.loc[0]

# • Примените TfidfVectorizer для преобразования текстов в век-
# торы признаков. Оставьте только те слова, которые встреча-
# ются хотя бы в 5 объектах.

vectorizer = TfidfVectorizer(min_df=5)
one_hot = DictVectorizer()

tf_idf = vectorizer.fit_transform(data.FullDescription) # count tf-idf for description

# • Замените пропуски в столбцах LocationNormalized и ContractTime
# на специальную строку ’nan’.

data.LocationNormalized.fillna('nan', inplace=True) # fill empty values in location and contract time
data.ContractTime.fillna('nan', inplace=True)

# • Примените DictVectorizer для получения one-hot-кодирования
# признаков LocationNormalized и ContractTime.

oh = one_hot.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records')) # one hot transformation for text fields

# • Объедините все полученные признаки в одну матрицу "объекты-признаки".

X = hstack([tf_idf, oh])
y = data['SalaryNormalized']

# 3. Обучите гребневую регрессию с параметром alpha=1. Целевая пе-
# ременная записана в столбце SalaryNormalized.

ridge = Ridge(alpha=1, random_state=241)
ridge.fit(X, y)

# 4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
# Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.

tf_idf_test = vectorizer.transform(test.FullDescription)
oh_test = one_hot.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([tf_idf_test, oh_test])

res = ridge.predict(X_test)

print res
f = open('4-1.txt', 'w')
f.write(str(res)[1:-1])
f.close()