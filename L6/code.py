# -*- coding: utf-8 -*-

from skimage.io import imread, imsave
from skimage import img_as_float
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from math import log10

def psnr(i1, i2):
    mse = ((i1 - i2) ** 2).mean()
    return 10 * log10(1. / mse)

# 1. Загрузите картинку parrots.jpg. Преобразуйте изображение, приве-
# дя все значения в интервал от 0 до 1.

img = imread('parrots.jpg')
img = img_as_float(img)
w, h = img.shape[:-1]
fimg = img.reshape(w*h, -1)


for n in range(1, 21):
    # 2. Создайте матрицу объекты-признаки: характеризуйте каждый пик-
    # сель тремя координатами - значениями интенсивности в простран-
    # стве RGB.
    X = pd.DataFrame(data=fimg, columns=('R', 'G', 'B'))

    # 3. Запустите алгоритм K-Means с параметрами init=’k-means++’ и
    # random_state=241. После выделения кластеров все пиксели, отне-
    # сенные в один кластер, попробуйте заполнить двумя способами:
    # медианным и средним цветом по кластеру.
    kmeans = KMeans(n_clusters=n, init='k-means++', random_state=241, n_jobs=-1)
    X['cluster'] = kmeans.fit_predict(X)
    means = X.groupby('cluster').mean().values
    medians = X.groupby('cluster').median().values
    set_new_colors = lambda a: np.reshape(map(lambda c: a[c], X['cluster']), (w, h, -1))
    mean_img = set_new_colors(means)
    median_img = set_new_colors(medians)
    imsave('mean_%d' % n + '.jpg', mean_img)
    imsave('median_%d' % n + '.jpg', median_img)

    # 4. Измерьте качество получившейся сегментации с помощью метрики PSNR.
    mean_psnr, median_psnr = psnr(img, mean_img), psnr(img, median_img)
    print n, mean_psnr, median_psnr

    # 5. Найдите минимальное количество кластеров, при котором значе-
    # ние PSNR выше 20 (можно рассмотреть не более 20 кластеров, но
    # не забудьте рассмотреть оба способа заполнения пикселей одного
    # кластера). Это число и будет ответом в данной задаче.
    if mean_psnr > 20 or median_psnr > 20:
        with open('6-1.txt', 'w') as f: f.write(str(n))
        break

