from os import listdir

import cv2
import numpy as np


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
    vol = 0
    for i in range(k):
        for j in range(l):
            vol += u[i][j] - b[i][j]
    return vol


def get_D(img):
    u1 = get_u(img)
    u2 = get_u(u1)
    u3 = get_u(u2)
    u4 = get_u(u3)
    u5 = get_u(u4)
    u6 = get_u(u5)

    b1 = get_b(img)
    b2 = get_b(b1)
    b3 = get_b(b2)
    b4 = get_b(b3)
    b5 = get_b(b4)
    b6 = get_b(b5)

    # vol1 = get_vol(u1, b1)
    # vol2 = get_vol(u2, b2)
    # vol3 = get_vol(u3, b3)
    vol4 = get_vol(u4, b4)
    vol5 = get_vol(u5, b5)
    vol6 = get_vol(u6, b6)

    A_2 = (vol5 - vol4) / (2)
    A_3 = (vol6 - vol5) / (2)

    return 2 - (np.log(A_2) - np.log(A_3)) / (np.log(5) - np.log(6))


def get_filepaths(dirpath):
    files = listdir(dirpath)
    for file in files:
        yield '%s/%s' % (dirpath, file), file


for path, filename in get_filepaths('../images'):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f'for {filename}: D={get_D(image)}')
