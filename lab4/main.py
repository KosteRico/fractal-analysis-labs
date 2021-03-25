from os import listdir

import cv2
import numpy as np

N = 200


def get_u(img):
    k, l = img.shape
    u = np.ndarray(img.shape)
    for i in range(k - 1):
        for j in range(l - 1):
            u[i][j] = max(img[i][j] + 1, max(img[i - 1][j], img[i][j - 1], img[i + 1][j], img[i][j + 1]))
    return u


def get_b(img):
    k, l = img.shape
    b = np.ndarray(img.shape)
    for i in range(k - 1):
        for j in range(l - 1):
            b[i][j] = min(img[i][j] - 1, min(img[i - 1][j], img[i][j - 1], img[i + 1][j], img[i][j + 1]))
    return b


def get_vol(u, b):
    k, l = u.shape
    vol = 0.0
    for i in range(k):
        for j in range(l):
            vol = vol + u[i][j] - b[i][j]
    return vol


def get_D(img):
    u1 = get_u(img)
    u2 = get_u(u1)
    u3 = get_u(u2)

    b1 = get_b(img)
    b2 = get_b(b1)
    b3 = get_b(b2)

    vol1 = get_vol(u1, b1)
    vol2 = get_vol(u2, b2)
    vol3 = get_vol(u3, b3)

    A_2 = (vol2 - vol1) / 2
    A_3 = (vol3 - vol2) / 2

    return 2 - (np.log(A_2) - np.log(A_3)) / (np.log(2) - np.log(3))


def get_filepaths(dirpath):
    files = listdir(dirpath)
    for file in files:
        yield '%s/%s' % (dirpath, file), file


for path, filename in get_filepaths('../images'):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f'{filename}: {get_D(img)}')
